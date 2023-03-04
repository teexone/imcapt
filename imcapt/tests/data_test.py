import json
import os
from imcapt.data.data import Flickr8DataModule
import torch as tr
import torchvision as travis
import numpy as np

dl = Flickr8DataModule(captions_path="./datasets/captions/dataset_flickr8k.json",
                       folder_path="./datasets/flickr8/images",
                       h5_load="./datasets/h5",
                       )
dl.setup(stage="")
img = json.load(open("./datasets/captions/dataset_flickr8k.json"))
img = img['images']
trans = travis.transforms.Resize((256, 256))

def test_train():
    count = 0
    ind = 0
    for input, label in dl.train_dataloader():
        count += 1
        if count % 250 == 0:
            print(f"Iters done: {count}")
        assert type(input) == tr.Tensor and type(label) == tr.Tensor
        data = []
        for i in input:
            data.append(i)
        data_pics = tr.stack(data)
        data_pics = data_pics.unique(dim=0)
        tst = []
        while len(tst) != len(data_pics):
            assert ind < len(img)
            if img[ind]['split'] == 'train' or img[ind]['split'] == 'restval':
                tenz = travis.io.read_image(os.path.join("./datasets/flickr8/Images/", img[ind]['filename']), mode=travis.io.ImageReadMode.RGB)
                tst.append(trans(tenz).to(tr.float32))
            ind += 1
        test_pics = tr.stack(tst)
        test_pics = test_pics.unique(dim=0)
        for t1, t2 in zip(data_pics, test_pics):
            assert tr.equal(t1, t2)
    print(f"Iter nums: {count}")

def test_test():
    count = 0
    ind = 0
    for input, label in dl.test_dataloader():
        count += 1
        if count % 250 == 0:
            print(f"Iters done: {count}")
        assert type(input) == tr.Tensor and type(label) == tr.Tensor
        data = []
        for i in input:
            data.append(i)
        data_pics = tr.stack(data)
        data_pics = data_pics.unique(dim=0)
        tst = []
        while len(tst) != len(data_pics):
            assert ind < len(img)
            if img[ind]['split'] == 'test':
                tenz = travis.io.read_image(os.path.join("./datasets/flickr8/Images/", img[ind]['filename']),
                                            mode=travis.io.ImageReadMode.RGB)
                tst.append(trans(tenz).to(tr.float32))
            ind += 1
        test_pics = tr.stack(tst)
        test_pics = test_pics.unique(dim=0)
        for t1, t2 in zip(data_pics, test_pics):
            assert tr.equal(t1, t2)

    print(f"Iter nums: {count}")


def test_val():
    count = 0
    ind = 0
    for input, label in dl.val_dataloader():
        count += 1
        if count % 250 == 0:
            print(f"Iters done: {count}")
        assert type(input) == tr.Tensor and type(label) == tr.Tensor
        data = []
        for i in input:
            data.append(i)
        data_pics = tr.stack(data)
        data_pics = data_pics.unique(dim=0)
        tst = []
        while len(tst) != len(data_pics):
            assert ind < len(img)
            if img[ind]['split'] == 'val':
                tenz = travis.io.read_image(os.path.join("./datasets/flickr8/Images/", img[ind]['filename']),
                                            mode=travis.io.ImageReadMode.RGB)
                tst.append(trans(tenz).to(tr.float32))
            ind += 1
        test_pics = tr.stack(tst)
        test_pics = test_pics.unique(dim=0)
        for t1, t2 in zip(data_pics, test_pics):
            assert tr.equal(t1, t2)
    print(f"Iter nums: {count}")