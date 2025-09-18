# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, pickle, six
from PIL import Image
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def target_to_oh(target: int):
    NUM_CLASS = 4  # hard code here, can do partial
    one_hot = [0,0,0,0]
    one_hot[target] = 1
    return one_hot


def build_dataset(is_train, args):
    transform = build_transform(is_train)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    root = args.data_path if is_train else args.eval_data_path
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 3
    assert len(dataset.class_to_idx) == nb_classes
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=(320, 320),
            is_training=True,
            auto_augment='rand',
            mean=mean,
            std=std,
        )
        return transform
    else:
        t = []
        t.append(transforms.ToTensor())
        t.append(transforms.Resize([320 ,320]))
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
