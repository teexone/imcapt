from collections import Counter, defaultdict
import copy
from typing import Dict, List
import pytorch_lightning as L
import torchvision as tvis
import torch as tr
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import os
import tqdm
import csv
import h5py


class Flickr8Dataset(Dataset):
    def __init__(self, images, captions) -> None:
        super().__init__()
        self._images = images
        self._captions = captions
        self._index = np.concatenate(([0], np.cumsum(list(map(len, captions)))[:-1],))
        # print(self._index)

    def _getindex(self, i):
        l, r = 0, len(self._index)
        while r - l > 1:
            m = (l + r) >> 1
            if self._index[m] <= i:
                l = m
            else:
                r = m
        return l

    def __getitem__(self, index) -> tr.Tensor:
        i = self._getindex(index)
        # assert i <= index
        # assert i < len(self._images), "image index out"
        # assert self._index[i] < len(self._captions), (len(self._captions), self._index[i],)
        # assert index - self._index[i] < len(self._captions[i])
        return self._images[i], tr.Tensor(self._captions[self._index[i]][index - self._index[i]])
    
    def __len__(self):
        return len(self._captions)

class Flickr8DataModule(L.LightningDataModule):
    DEFAULT_TRANSFORMS = tr.nn.Sequential(tvis.transforms.Resize((256, 256,)))

    def __init__(self, 
                *args, 
                 folder_path=None, 
                 captions_path=None, 
                 word_map_path=None,
                 h5_load=None,
                 force_word_map_update=False,
                 batch_size=20, 
                 max_caption_length=100, 
                 transforms=DEFAULT_TRANSFORMS, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self._image_folder_path = folder_path
        self._captions_json = captions_path

        self._splits = ["train", "test", "val"]

        self.images = defaultdict(lambda: h5py.Empty(np.float32))
        self.captions = defaultdict(lambda: h5py.Empty(np.float32))
        self.word_map = h5py.Empty(np.float32)

        self.transforms = transforms
        self.batch_size = batch_size
        self.max_caption_length = max_caption_length

        self._word_map_path = word_map_path if word_map_path is not None else "./word_map.csv" 
        self._force_word_map_update = force_word_map_update
        self._h5_load_path = h5_load

    def _h5load(self, verbose=False):
        file = h5py.File(os.path.join(self._h5_load_path, "flickr8.hdf5"), "r")
        images, captions, wmap = {}, {}, json.loads(file.get(f"word_map")[0]) 
        for split in self._splits:
            images[split] = file.get(f"{split}/images")
            captions[split] = file.get(f"{split}/captions")
        return images, captions, wmap

    def _h5save(self, images, encoded_captions, wordmap):
        assert self._h5load is not None
        file = h5py.File(os.path.join(self._h5_load_path, "flickr8.hdf5"), "a")
        for split in images.keys():
            try:
                file.create_group(split)
            except ValueError:
                continue
    
        for split, data in images.items():
            data = tr.stack(data)
            dataset = file.require_dataset(f"{split}/images", shape=data.size(), dtype=np.float32)
            dataset[:] = data
        for split, data in encoded_captions.items():
            data = tr.FloatTensor(data)
            data = tr.cat(data.unbind(dim=0))
            print(data.size())
            dataset = file.require_dataset(f"{split}/captions", data=data, shape=data.size(), dtype=np.float32)
            dataset[:] = data
        dataset = file.require_dataset("word_map", shape=(1,), dtype=h5py.string_dtype())
        dataset[0] = json.dumps(wordmap)


    def setup(self, stage, verbose=False) -> None:
        super().setup(stage)
        captions_dict = json.load(open(self._captions_json))
        
        if self._h5_load_path is not None:
            try:
                self.images, self.captions, self.word_map = self._h5load()
                return
            except Exception as e:
                if verbose:
                    print("Exception occured when trying to load the data. Computing the data...")

        
        words = set()

        captions = defaultdict(lambda: [])
        images = defaultdict(lambda: [])

        for image in tqdm.tqdm(captions_dict['images'], desc="Reading files...", disable=verbose, bar_format="{desc} | {bar} | {n_fmt} of {total}"):    
            image_captions = []
            for caption in image['sentences']:
                if len(caption['tokens']) <= self.max_caption_length:
                    image_captions.append(caption['tokens'])
                    words.update(caption['tokens'])
            if not image_captions:
                print("skip")
                continue

            image_file_name = image['filename']
            image_file_path = os.path.join(self._image_folder_path, image_file_name)
            tensor_image = tvis.io.read_image(image_file_path)
            tensor_image = self.transforms(tensor_image)

            split = image['split'] if image['split'] != 'restval' else 'train'
            captions[split].append(image_captions)
            images[split].append(tensor_image.to(tr.float32))

        if verbose:
            print("Data loading. Scanning the word map...")

        word_map = self._create_word_map(words)

        encoded_captions = defaultdict(lambda: [])
        for spl, data in tqdm.tqdm(captions.items(), desc="Encoding captions...", bar_format="{desc} | {bar} | {n_fmt} of {total}", disable=verbose):
            for image_captions in copy.deepcopy(data):
                encoded_sentences = []
                for sentence in image_captions:
                    encoded = [word_map['<BGN>']] +\
                            [word_map.get(word, word_map['<WTF>']) for word in sentence] +\
                            [word_map['<END>']] +\
                            [word_map['<PAD>']] * (self.max_caption_length - len(sentence) - 2) 
                    assert len(encoded) == self.max_caption_length, len(encoded)
                    encoded_sentences.append(encoded)
                                    
                encoded_captions[spl].append(encoded_sentences)
        
        # self.images = images
        # self.captions = encoded_captions

        if self._h5_load_path is not None:
            self._h5save(images, encoded_captions, word_map)

        self.images, self.captions, self.word_map = self._h5load()

    
    def _create_word_map(self, words=None):
        word_map = {k: i for i, k in enumerate(words, start=1)}
        word_map['<BGN>'] = len(word_map)
        word_map['<END>'] = len(word_map)
        word_map['<WTF>'] = len(word_map)
        word_map['<PAD>'] = 0
        return word_map
    
    def _save_word_map(self, word_map):
        csv = csv.writer(open(self._word_map_path, "w+"), delimeter=",")
        for word, embedding in word_map.items():
            csv.writerow([word, embedding])


        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=Flickr8Dataset(images=self.images['train'], 
                                                 captions=self.captions['train']), 
                                                 batch_size=self.batch_size)
    
    def val_dataloader(self) -> DataLoader:

        return DataLoader(dataset=Flickr8Dataset(self.images['val'], 
                                                self.captions['val']), 
                                                batch_size=self.batch_size)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=Flickr8Dataset(self.images['test'], 
                                                self.captions['test']), 
                                                batch_size=self.batch_size)
                

if __name__ == "__main__":
    dl = Flickr8DataModule(captions_path="./datasets/captions/dataset_flickr8k.json", 
                           folder_path="./datasets/flickr8/images",
                           h5_load="./datasets/h5")
    dl.setup(stage="")
    count = 0
    # for input, label in dl.train_dataloader():
        
    print(len(dl.train_dataloader().dataset._captions))
        