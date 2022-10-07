## Okay working

from dataset import get_dataset

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform_train = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.444, 0.421, 0.385), 
                                 (0.285, 0.277, 0.286))])

transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize((0.444, 0.421, 0.385), 
                                 (0.285, 0.277, 0.286))])

def collate_custom(batch):
 
    imgs = torch.stack([item[0] for item in batch])
    caps = [item[1] for item in batch]
 
    return imgs, caps

def get_dataloader(batch_size,shuffle,num_workers,split):
 
    if split == "train":
        dataset = get_dataset(transform=transform_train,split="train")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collate_custom)
    elif split == "test":
        dataset = get_dataset(transform=transform_test,split="test")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collate_custom)
    else:
        print("Wrong split")