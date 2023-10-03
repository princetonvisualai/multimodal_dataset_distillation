
import torch
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10
from collections import defaultdict


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PROMPTS_1 = ["This is a {}"]

PROMPTS_5 = [
    "a photo of a {}",
    "a blurry image of a {}",
    "a photo of the {}",
    "a pixelated photo of a {}",
    "a picture of a {}",
]

PROMPTS = [
    'a photo of a {}',
    'a blurry photo of a {}',
    'a low contrast photo of a {}',
    'a high contrast photo of a {}',
    'a bad photo of a {}',
    'a good photo of a {}',s
    'a photo of a small {}',
    'a photo of a big {}',
    'a photo of the {}',
    'a blurry photo of the {}',
    'a low contrast photo of the {}',
    'a high contrast photo of the {}',
    'a bad photo of the {}',
    'a good photo of the {}',
    'a photo of the small {}',
    'a photo of the big {}',
]


class cifar10_train(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num_prompts=1):
        super(cifar10_train, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if num_prompts == 1:
            self.prompts = PROMPTS_1
        elif num_prompts == 5:
            self.prompts = PROMPTS_5
        else:
            self.prompts = PROMPTS
        self.captions = [prompt.format(cls) for cls in CLASSES for prompt in self.prompts]
        self.captions_to_label = {cap: i for i, cap in enumerate(self.captions)}
        self.annotations = []
        for i in range(len(self.data)):
            cls_name = CLASSES[self.targets[i]]
            for prompt in self.prompts:
                caption = prompt.format(cls_name)
                self.annotations.append({"img_id": i, "caption_id": self.captions_to_label[caption]})
        if num_prompts == 1:
            self.annotations = self.annotations * 5
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        ann = self.annotations[index]
        img_id = ann['img_id']
        img = self.transform(self.data[img_id])
        caption = self.captions[ann['caption_id']]
        return img, caption, img_id

    def fetch_distill_images(self, ipc):
        """
        Randomly fetch `x` number of images from each class using numpy and return as a tensor.
        """
        class_indices = defaultdict(list)

        for idx, label in enumerate(self.targets):
            class_indices[label].append(idx)

        # Randomly sample x indices for each class using numpy
        sampled_indices = [np.random.choice(indices, ipc, replace=False) for indices in class_indices.values()]
        sampled_indices = [idx for class_indices in sampled_indices for idx in class_indices]

        # Fetch images and labels using the selected indices
        images = torch.stack([self.transform(self.data[i]) for i in sampled_indices])
        labels = [self.targets[i] for i in sampled_indices]

        captions = []
        for label in labels:
            cls_name = CLASSES[label]
            prompt = np.random.choice(self.prompts)
            random_caption = prompt.format(cls_name)
            captions.append(random_caption)

        return images, captions

    def get_all_captions(self):
        return self.captions


class cifar10_retrieval_eval(cifar10_train):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, num_prompts=1):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super(cifar10_retrieval_eval, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download, num_prompts=num_prompts)
        self.text = self.captions
        self.txt2img = {}
        self.img2txt = defaultdict(list)

        for ann in self.annotations:
            img_id = ann['img_id']
            caption_id = ann['caption_id']
            self.img2txt[img_id].append(caption_id)
            self.txt2img[caption_id] = img_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.transform(self.data[index])
        return image, index
