## Okay working

from dataset import get_dataset
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


transform_train = transforms.Compose([
            transforms.RandomCrop(224,pad_if_needed=True),
            transforms.RandomHorizontalFlip(), 
            ])

transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            ])





def get_dataloader(tokenizer,feature_extractor,rank,world_size,batch_size,shuffle,num_workers,split,pin_memory=False):
    
    
    if split == "train":
        
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=num_workers,sampler=sampler,drop_last=True)
    
    elif split == "val":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=num_workers,sampler=sampler,drop_last=False)
    
    elif split == "test":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=num_workers,sampler=sampler,drop_last=False)
    else:
        print("Wrong split")

# TODO Remove once MOCO works

'''

def get_dataloader(tokenizer,feature_extractor,batch_size,shuffle,num_workers,split):
    
    
    if split == "train":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)
    elif split == "val":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=False)
    elif split == "test":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=False)
    else:
        print("Wrong split")'''