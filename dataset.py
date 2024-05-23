import re
import json
import os
from PIL import Image
import random 
import os.path as osp
import numpy as np
import torch
from torch.utils.data import  Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms

from utils import read_json
import config as CFG

import webdataset as wds

#############################################################################
#                                                                           #
#                              Flicker30k                                   #
#                                                                           #
#############################################################################

def pre_caption(caption,max_words=CFG.max_length):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class flickr30k(Dataset):
    def __init__(self, tokenizer,feature_extractor,transform,image_root, ann_root, split, max_words=CFG.max_length, prompt=CFG.prompt):        
        '''
        image_root (string): Root directory of images (e.g. data/)
        ann_root (string): directory to store the annotation file
        split (string): one of "train" or "test"
        '''        
      

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.split = split
        assert self.split in ("train","val","test")

        if self.split == "train":
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json"
            filename = "flickr30k_train.json"
        elif self.split == "val":
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json"
            filename = "flickr30k_val.json"
        else:
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json"
            filename = "flickr30k_test.json"

        #download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        
        self.img_ids = {} 
        
        ## Totally bad way to merge the caption, cba to redo

        if self.split == "train":
            n = 0
            last_img_id = None
            current_img_id = None
            for ann in self.annotation:
                current_img_id = ann["image_id"]

                if current_img_id != last_img_id:
                    self.img_ids[n] = ann
                    self.img_ids[n]["caption"] = [self.img_ids[n]["caption"]]
                    last_img_id = current_img_id
                    n += 1
                else: 
                    ls = self.img_ids[n-1]["caption"]
                    ls.append(ann["caption"])               
                    self.img_ids[n-1]["caption"] = ls
                
                
        else:
            n = 0
            for ann in self.annotation:
                self.img_ids[n] = ann
                n += 1

        self.annotation = None
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):    
        
        item = self.img_ids[index]
        
        image_path = os.path.join(self.image_root,item['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(random.choice(item['caption']), self.max_words)
        
        #for CLIP
        #image_encoded = self.feature_extractor(text=None,images=image,return_tensors="pt")
        # For our
        image_encoded = self.feature_extractor(image,return_tensors="pt")

        caption_encoded = self.tokenizer(caption,padding="max_length",max_length=self.max_words)

        return {"image" :image_encoded["pixel_values"].squeeze(0), "input_ids": torch.as_tensor(caption_encoded["input_ids"]), "attention_mask": torch.as_tensor(caption_encoded["attention_mask"])}


#############################################################################
#                                                                           #
#                                ms COCO                                    #
#                                                                           #
#############################################################################

def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions


class mscoco(Dataset):
    def __init__(self, tokenizer,feature_extractor,transform,image_root, ann_root, split, max_words=CFG.max_length, prompt=CFG.prompt):        
        '''
        image_root (string): Root directory of images (e.g. data/)
        ann_root (string): directory to store the annotation file
        split (string): one of "train" or "val"
        '''        
      

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.split = split
        assert self.split in ("train","val")

        if self.split == "train":
            filename = "captions_train2017.json"
            image_split = "train2017"
        elif self.split == "val":
            filename = "captions_val2017.json"
            image_split = "val2017"
                
        annotations = read_json(os.path.join(ann_root,filename))
        self.transform = transform
        self.image_root = os.path.join(image_root,image_split)
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):    
        
        img_id = self.img_ids[index]
        
        img_filename = self.img_id_to_filename[img_id]
        image_path = os.path.join(self.image_root,img_filename)        
        image = Image.open(image_path).convert('RGB')   
        
        image = self.transform(image)
        
        caption = self.prompt+random.choice(self.img_id_to_captions[img_id])
        
        #for CLIP
        #image_encoded = self.feature_extractor(text=None,images=image,return_tensors="pt")
        # For our
        
        image_encoded = self.feature_extractor(image,return_tensors="pt")

        caption_encoded = self.tokenizer(caption,padding="max_length",max_length=self.max_words)

        return {"image" :image_encoded["pixel_values"].squeeze(0), "input_ids": torch.as_tensor(caption_encoded["input_ids"]), "attention_mask": torch.as_tensor(caption_encoded["attention_mask"])}


#############################################################################
#                                                                           #
#                                  cc3m                                     #
#                                                                           #
#############################################################################

class cc3mDataset(wds.WebDataset):
    def __init__(self, tokenizer,feature_extractor,url):
        #super().__init__(url)
        self.dataset = wds.WebDataset(url).shuffle(1000).decode("rgb")
        
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def __iter__(self):
        
        for item in self.dataset:
            yield self.process_item(item)
        

    def process_item(self, item):

        caption_encoded = self.tokenize_caption(item["txt"])
        image_encoded = self.extract_features(item["jpg"])
        
        return {"image" :image_encoded["pixel_values"].squeeze(0), "input_ids": torch.as_tensor(caption_encoded["input_ids"]), "attention_mask": torch.as_tensor(caption_encoded["attention_mask"])}

    def tokenize_caption(self, caption):
        caption_encoded = self.tokenizer(caption,padding="max_length",max_length=CFG.max_length)
        return caption_encoded

    def extract_features(self, image):
        image_tensor = self.transform(image)
        image_tensor = image_tensor  # Add batch dimension
        image_features = self.feature_extractor(image_tensor,return_tensors="pt")
        return image_features

########################################################################################


def get_dataset(dataset,tokenizer,feature_extractor,transform,split,rank=0):
    
    if dataset == "flickr30k":
        image_root=""
        ann_root="./flickr30k"
        return flickr30k(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform,image_root=image_root,ann_root=ann_root,split=split)
    elif dataset == "mscoco":
        image_root = "data/mscoco"
        ann_root="data/mscoco/annotations"
        return mscoco(tokenizer,feature_extractor,transform,image_root,ann_root,split)
    elif dataset == "cc3m":
        assert split in ("train","val")
        if split == "train":
            if rank == 0:
                url = "data/cc3m/train/{00000..00165}.tar"
                #url = "data/cc3m/train/{00000..00001}.tar"
            else:
                url = "data/cc3m/train/{00166..00331}.tar"
        elif split == "val":
            if rank == 0:
                url = "data/cc3m/val/00000.tar"
            else:
                url = "data/cc3m/val/00001.tar"
        return cc3mDataset(tokenizer,feature_extractor,url)
    elif dataset == "ade":
        ade_root = "data/ade/ADEChallengeData2016"
        return ADE20KDataset(feature_extractor,transform,ade_root,split)
    




#############################################################################
#                                                                           #
#                                ADE 20k                                    #
#                                                                           #
#############################################################################



class ADE20KDataset(Dataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    def __init__(self,feature_extractor,transform,ade_root,split):
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.split = split
        assert self.split in ("train","val")

        if self.split == "train":
            image_split = "train"
        elif self.split == "val":
            image_split = "validation"
        
        self.image_root = os.path.join(ade_root,"images",image_split)
        self.ann_root = os.path.join(ade_root,"annotations",image_split)


        # load annotations
        self.img_infos = self.load_annotations(self.image_root,self.ann_root)


    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_root,ann_root):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        for file in os.listdir(img_root):
            
            img_name = os.path.splitext(file)[0]
            img_info = dict(filename=img_name + ".jpg")
            
            seg_map = img_name + ".png"
            img_info['ann'] = dict(seg_map=seg_map)
            
            img_infos.append(img_info)
        
        print(f'Loaded {len(img_infos)} images')
        return img_infos      

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']


    def __getitem__(self, idx):
        img_path = self.img_infos[idx]["filename"]
        seg_path = self.img_infos[idx]["ann"]["seg_map"]
        
        img_path = os.path.join(self.image_root,img_path)
        seg_path = os.path.join(self.ann_root,seg_path)

       
        image = Image.open(img_path).convert('RGB')   
        
        image = self.transform(image)
        image_encoded = self.feature_extractor(image,return_tensors="pt")
        seg_map = Image.open(seg_path).convert('RGB')   
        
        seg_map = self.transform(seg_map)

        seg_map_encoded = np.asarray(seg_map)
        
        return {"image" :image_encoded["pixel_values"].squeeze(0),"seg_map":seg_map_encoded}

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
