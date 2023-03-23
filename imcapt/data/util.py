import json
import h5pickle
import torch

from imcapt.data.vocabulary import Vocabulary


def load_from_h5(path, splits=["train", "val", "test"]):
    file = h5pickle.File(path, "r")
    images, captions, captions_iids, vocabular = {}, {}, {}, Vocabulary.from_h5(file)  
    for split in splits:
        images[split] = file.get(f"{split}/images")
        captions[split] = file.get(f"{split}/captions")
        captions_iids[split] = file.get(f"{split}/captions_iids")
    return images, captions, captions_iids, vocabular

def extract_vocabulary_from_karpathy_json(file_path, max_caption_length=40):
    captions_dict = json.load(open(file_path))
    vocabulary = Vocabulary()
    for image in captions_dict['images']:
        for caption in image['sentences']:
            if len(caption['tokens']) + 2 <= max_caption_length:
                vocabulary.update(caption['tokens'])
    return vocabulary

