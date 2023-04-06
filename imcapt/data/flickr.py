import copy
from collections import defaultdict
import pytorch_lightning as L
import torchvision as tvis
import torch as torch
import numpy as np
import json
import os
import tqdm
import h5py
import h5pickle
from torch.utils.data import DataLoader, Dataset
from multiprocessing import cpu_count
from ..data.vocabulary import Vocabulary


DEFAULT_TRANSFORMS = torch.nn.Sequential(tvis.transforms.Resize((256, 256,)))


def prepare_h5(
            captions_path: os.PathLike, 
            image_folder_path: os.PathLike, 
            h5_file_path: os.PathLike, 
            max_caption_length=40,
            verbose=True
        ):
        """Prepares HDF file with Flickr dataset."""
        vocabulary = Vocabulary()
        captions_dict = json.load(open(captions_path))
        captions = defaultdict(lambda: [])
        images = defaultdict(lambda: [])

        reading_files_pgbar = tqdm.tqdm(
            captions_dict['images'], 
            desc="Reading files...", 
            disable=not verbose, 
            bar_format="{desc} | {bar} | {n_fmt} of {total}")

        captions_count = 0
        for i, image in enumerate(reading_files_pgbar):   
            image_captions = []
            for caption in image['sentences']:
                if len(caption['tokens']) + 2 <= max_caption_length:
                    captions_count += 1
                    image_captions.append(caption['tokens'])
                    vocabulary.update(caption['tokens'])

            if not image_captions:
                continue

            image_file_name = image['filename']
            image_file_path = os.path.join(image_folder_path, image_file_name)

            tensor_image = tvis.io.read_image(image_file_path)
            tensor_image = DEFAULT_TRANSFORMS(tensor_image)

            split = image['split'] if image['split'] != 'restval' else 'train'
            captions[split].append((len(images[split]), image_captions,))
            images[split].append(tensor_image.to(torch.float32))


        encoded_captions = defaultdict(lambda: [])
        encoded_captions_iids = defaultdict(lambda: [])

        encoding_pgbar = tqdm.tqdm(desc="Encoding captions...", 
                                   bar_format="{desc} | {bar} | {n_fmt} of {total}", 
                                   disable=not verbose,
                                   total=captions_count)
    
        for spl, data in captions.items():
            for i, image_captions in copy.deepcopy(data):
                encoded_sentences = []
                encoded_sentences_iids = []
                for sentence in image_captions:
                    encoded = [vocabulary['<START>']] +\
                            [vocabulary[word] for word in sentence] +\
                            [vocabulary['<END>']] +\
                            [vocabulary['<PADDING>']] * (max_caption_length - len(sentence) - 2) 
                    assert len(encoded) == max_caption_length, len(encoded)
                    encoded_sentences.append(encoded) 
                    encoded_sentences_iids.append(i)
                    encoding_pgbar.update(1)
                encoded_captions[spl].extend(encoded_sentences)
                encoded_captions_iids[spl].extend(encoded_sentences_iids)
        encoding_pgbar.close()
        for spl in captions.keys():
            assert len(encoded_captions_iids[spl]) == len(encoded_captions[spl])
            assert max(encoded_captions_iids[spl]) < len(images[spl])

        file = h5py.File(h5_file_path, "a")

        for split in images.keys():
            try:
                file.create_group(split)
            except ValueError:
                continue

            
        for split, data in images.items():
            dataset = file.require_dataset(f"{split}/images", shape=(len(data), *data[0].size()), dtype=np.float32)
            for i, t in enumerate(data):
                # data = torch.stack(data)
                dataset[i] = t
        for split, data in encoded_captions.items():
            data = torch.LongTensor(data)
            dataset = file.require_dataset(f"{split}/captions",  shape=data.size(), dtype=np.float32)
            dataset[:] = data
        for split, data in encoded_captions_iids.items():
            data = torch.LongTensor(data)
            dataset = file.require_dataset(f"{split}/captions_iids",  shape=data.size(), dtype=np.float32)
            dataset[:] = data
        
        vocabulary.to_h5(file)
        


class FlickrDataset(Dataset):
    def __init__(
            self, 
            images,
            captions, 
            captions_iids,
            reduce_samples_to=-1, 
            combine=False
        ) -> None:
        super().__init__()
        self._images = images
        self._captions = captions
        self._captions_iids = captions_iids
        self._combined = combine
        self._reduce_samples_to = reduce_samples_to
        
        if combine:
            self.__combined = defaultdict(list)
            max_size = 0
            for c, i in zip(self._captions, self._captions_iids):
                self.__combined[i].append(torch.LongTensor(c))
                max_size = max(max_size, len(self.__combined[i]))
            
            for _c in self.__combined.values():
                while len(_c) < max_size:
                    _c.append(_c[-1])
    
    def __getitem__(self, index) -> torch.Tensor:
        if self._combined:
            return self._images[index], torch.stack(self.__combined[index])
        else:
            return self._images[int(self._captions_iids[index])], torch.LongTensor(self._captions[index])
    
    def __len__(self):
        if self._reduce_samples_to != -1:
            return self._reduce_samples_to
        
        if self._combined:
            return len(self._images)
        else:
            return len(self._captions)

class FlickrDataModule(L.LightningDataModule):
    
    def __init__(self, 
                *args, 
                 folder_path=None, 
                 captions_path=None, 
                 h5_load=None,
                 batch_size=20, 
                 max_caption_length=100,
                 reduce_train_samples_to=-1,
                 vocabular=Vocabulary(),
                 transforms=DEFAULT_TRANSFORMS, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self._image_folder_path = folder_path
        self._captions_json = captions_path
        self._default_verbose=False

        if folder_path is not None and not os.path.exists(folder_path):
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        if captions_path is not None and not os.path.exists(captions_path):
            os.makedirs(os.path.dirname(captions_path), exist_ok=True)

        if h5_load is not None and not os.path.exists(h5_load):
            os.makedirs(os.path.dirname(h5_load), exist_ok=True)

        self._splits = ["train", "test", "val"]

        self.images = defaultdict(lambda: h5py.Empty(np.float32))
        self.captions = defaultdict(lambda: h5py.Empty(np.float32))
        self.caption_iids = defaultdict(lambda: h5py.Empty(np.int32))

        self.vocabulary = vocabular
        self.transforms = transforms
        self.batch_size = batch_size
        self.max_caption_length = max_caption_length

        self._h5_path = h5_load
        self._preload = False
        self._reduce_train_sample = reduce_train_samples_to

    def prepare_data(self) -> None:
        try:
            self.images, self.captions, self.caption_iids, self.vocabulary = self._h5load()
            self._preload = True
        except:
            ...


    def _h5load(self, verbose=True):
        file = h5pickle.File(self._h5_path, "r")
        images, captions, captions_iids, vocabular = {}, {}, {}, Vocabulary.from_h5(file)  
        for split in self._splits:
            images[split] = file.get(f"{split}/images")
            captions[split] = file.get(f"{split}/captions")
            captions_iids[split] = file.get(f"{split}/captions_iids")
        return images, captions, captions_iids, vocabular


    def _h5save(self, images, encoded_captions, encoded_caption_iids):
        file = h5py.File(self._h5_path, "a")
        for split in images.keys():
            try:
                file.create_group(split)
            except ValueError:
                continue

        
        for split, data in images.items():
            dataset = file.require_dataset(f"{split}/images", shape=(len(data), *data[0].size()), dtype=np.float32)
            for i, t in enumerate(data):
                # data = torch.stack(data)
                dataset[i] = t
        for split, data in encoded_captions.items():
            data = torch.LongTensor(data)
            dataset = file.require_dataset(f"{split}/captions",  shape=data.size(), dtype=np.float32)
            dataset[:] = data
        for split, data in encoded_caption_iids.items():
            data = torch.LongTensor(data)
            dataset = file.require_dataset(f"{split}/captions_iids",  shape=data.size(), dtype=np.float32)
            dataset[:] = data

        self.vocabulary.to_h5(file)


    def initialize(self, verbose=True):
        captions_dict = json.load(open(self._captions_json))
        if self._h5_path is not None and self._preload:
            return
        elif not self._preload:
            self.prepare_data()
            if self._preload: return
        
        captions = defaultdict(lambda: [])
        images = defaultdict(lambda: [])

        reading_files_pgbar = tqdm.tqdm(
            captions_dict['images'], 
            desc="Reading files...", 
            disable=not verbose, 
            bar_format="{desc} | {bar} | {n_fmt} of {total}")

        captions_count = 0
        for i, image in enumerate(reading_files_pgbar):   
            image_captions = []
            for caption in image['sentences']:
                if len(caption['tokens']) + 2 <= self.max_caption_length:
                    captions_count += 1
                    image_captions.append(caption['tokens'])
                    self.vocabulary.update(caption['tokens'])

            if not image_captions:
                continue

            image_file_name = image['filename']
            image_file_path = os.path.join(self._image_folder_path, image_file_name)

            tensor_image = tvis.io.read_image(image_file_path)
            tensor_image = self.transforms(tensor_image)

            split = image['split'] if image['split'] != 'restval' else 'train'
            captions[split].append((len(images[split]), image_captions,))
            images[split].append(tensor_image.to(torch.float32))


        encoded_captions = defaultdict(lambda: [])
        encoded_captions_iids = defaultdict(lambda: [])

        encoding_pgbar = tqdm.tqdm(desc="Encoding captions...", 
                                   bar_format="{desc} | {bar} | {n_fmt} of {total}", 
                                   disable=not verbose,
                                   total=captions_count)
    
        for spl, data in captions.items():
            for i, image_captions in copy.deepcopy(data):
                encoded_sentences = []
                encoded_sentences_iids = []
                for sentence in image_captions:
                    encoded = [self.vocabulary['<START>']] +\
                            [self.vocabulary[word] for word in sentence] +\
                            [self.vocabulary['<END>']] +\
                            [self.vocabulary['<PADDING>']] * (self.max_caption_length - len(sentence) - 2) 
                    assert len(encoded) == self.max_caption_length, len(encoded)
                    encoded_sentences.append(encoded) 
                    encoded_sentences_iids.append(i)
                    encoding_pgbar.update(1)
                encoded_captions[spl].extend(encoded_sentences)
                encoded_captions_iids[spl].extend(encoded_sentences_iids)
        encoding_pgbar.close()
        for spl in captions.keys():
            assert len(encoded_captions_iids[spl]) == len(encoded_captions[spl])
            assert max(encoded_captions_iids[spl]) < len(images[spl])
        if self._h5_path is not None:
            self._h5save(images, encoded_captions, encoded_captions_iids)

        self.images, self.captions, self.caption_iids, self.vocabulary = self._h5load()

        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=FlickrDataset(images=self.images['train'], 
                                                 captions=self.captions['train'],
                                                 reduce_samples_to=self._reduce_train_sample,
                                                 captions_iids=self.caption_iids['train']), 
                                                 batch_size=self.batch_size,
                                                 num_workers=cpu_count(),
                                                 drop_last=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=FlickrDataset(self.images['val'], 
                                                self.captions['val'],
                                                self.caption_iids['val'],
                                                combine=True),
                                                batch_size=self.batch_size,
                                                drop_last=True)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=FlickrDataset(self.images['test'], 
                                                self.captions['test'],
                                                self.caption_iids['test']), 
                                                batch_size=self.batch_size,
                                                drop_last=True)
                

    def setup(self, stage, verbose=False) -> None:
        super().setup(stage)
        self.initialize(verbose)
        
       