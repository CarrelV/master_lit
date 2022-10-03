import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 16
download_img = True
shuffle_img = True
num_workers = 2

dataset = torchvision.datasets.Flickr30k(root='./data',ann_file="flickr30k_train.json", 
                                         transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=shuffle_img, num_workers=num_workers)
