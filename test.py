import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging
import torchvision.transforms as transforms
#from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from robustness.tools.imagenet_helpers import common_superclass_wnid,ImageNetHierarchy
from robustness import datasets
from models_CLIP import CLIPMoco
#from models import CLIPMoco
import warnings
import csv 

import config as CFG
from tokenizer import get_tokenizer,get_feature_extractor
from dataloader import get_dataloader
from utils import read_imagenet_class
import os
data_to_save = []

def test():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Start testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()

    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("test 1")
    print(CFG.image_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)
    print("test 2")
    print(CFG.text_model_name)
    tokenizer = get_tokenizer(CFG.text_model_name)

    print("Hallo")
    model = CLIPMoco().to(device)

    data_to_save.append(CFG.run_info)
    data_to_save.append(CFG.training_dataset)
    data_to_save.append(CFG.text_tower_name)
    data_to_save.append(CFG.text_head_name)
    data_to_save.append(CFG.image_tower_name)
    data_to_save.append(CFG.image_head_name)

    # Load the weights for the backbone

    print("Loading the saved weights")
    if CFG.text_backbone_finetune:
        model.text_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_enc_{CFG.version_add_information}_{CFG.weight_version}.pt",map_location=device))
        data_to_save.append(f"{CFG.configuration_to_test}_text_enc_{CFG.version_add_information}_{CFG.weight_version}.pt")
    else:
        data_to_save.append(CFG.text_model_name)    

    data_to_save.append(f"{CFG.configuration_to_test}_text_proj_{CFG.version_add_information}_{CFG.weight_version}.pt")

    if CFG.image_backbone_finetune:
        model.image_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_enc_{CFG.version_add_information}_{CFG.weight_version}.pt",map_location=device))
        data_to_save.append(f"{CFG.configuration_to_test}_img_enc_{CFG.version_add_information}_{CFG.weight_version}.pt")
    else:
        data_to_save.append(CFG.image_model_name)
        

    # Load the heads
    model.text_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_proj_{CFG.version_add_information}_{CFG.weight_version}.pt",map_location=device))
    model.image_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_proj_{CFG.version_add_information}_{CFG.weight_version}.pt",map_location=device))
    data_to_save.append(f"{CFG.configuration_to_test}_img_proj_{CFG.version_add_information}_{CFG.weight_version}.pt")
    '''
    #testing with a pretrained clip
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    feature_extractor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    '''
    print("Model loaded")
    
    ## Number of params

    text_tower_trainable_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    text_tower_total_params = sum(p.numel() for p in model.text_encoder.parameters())
    text_head_trainable_params = sum(p.numel() for p in model.text_projection.parameters() if p.requires_grad)
    text_head_total_params = sum(p.numel() for p in model.text_projection.parameters())
    image_tower_trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    image_tower_total_params = sum(p.numel() for p in model.image_encoder.parameters())
    image_head_trainable_params = sum(p.numel() for p in model.image_projection.parameters() if p.requires_grad)
    image_head_total_params = sum(p.numel() for p in model.image_projection.parameters())

    data_to_save.append(text_tower_total_params)
    data_to_save.append(text_head_total_params)
    data_to_save.append(image_tower_total_params)
    data_to_save.append(image_head_total_params)
    data_to_save.append(text_tower_trainable_params)
    data_to_save.append(text_head_trainable_params)
    data_to_save.append(image_tower_trainable_params)
    data_to_save.append(image_head_trainable_params)
    ## End Number of params


    model.eval()

    print("-------------------------")
    print(f"For the model {CFG.configuration_to_test}, weights: {CFG.configuration_to_test}_{CFG.version_add_information}_{CFG.weight_version}, text model size: {CFG.text_model_size}, image model size: {CFG.image_model_size}")
    print("-------------------------\n")
    


    # to stop, only when testing the loading in local
    
    

    print("0 Shot classification on ImageNetV2:")

    top_1,top_5 = imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,dataset = "all",device=device)
    data_to_save.append(top_1)
    data_to_save.append(top_5)
    top_1,top_5 = imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,dataset = "big",device=device)
    data_to_save.append(top_1)
    data_to_save.append(top_5)
    top_1,top_5 = imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,dataset = "medium",device=device)
    data_to_save.append(top_1)
    data_to_save.append(top_5)
    top_1,top_5 = imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,dataset = "small",device=device)
    data_to_save.append(top_1)
    data_to_save.append(top_5)
    top_1,top_5 = imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,dataset = "tiny",device=device)
    data_to_save.append(top_1)
    data_to_save.append(top_5)

    print("T2I and I2T retrieval on Flickr30k test set:")

    top1_i2t,top5_i2t,top10_i2t,top1_t2i,top5_t2i,top10_t2i = i2t_t2i_retrieval(model=model,dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)
    data_to_save.append(top1_i2t)
    data_to_save.append(top5_i2t)
    data_to_save.append(top10_i2t)
    data_to_save.append(top1_t2i)
    data_to_save.append(top5_t2i)
    data_to_save.append(top10_t2i)
    #No test set for mscoco
    #i2t_t2i_retrieval(model=model,dataset="mscoco",tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)

    with open("results.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_to_save)



######################## IMAGENET 0 SHOT ###############
def imagenet_0shot(model,tokenizer,feature_extractor,dataset,device):

    imagenet_prompt ='a photo of a {}.'


    transform_ImageNet = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.444, 0.421, 0.385), 
                                     (0.285, 0.277, 0.286)),
                ])
    
    print("-------------------------\n")

    if dataset == "all":
        

        ds = ImageNetV2Dataset(transform=transform_ImageNet)
        test_loader = DataLoader(ds, batch_size=CFG.test_batch_size, num_workers=CFG.num_workers)

        imagenet_classes = read_imagenet_class()
            

        print(f"On the complete test set with {len(imagenet_classes)} different classes")

    elif dataset == "tiny":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid = common_superclass_wnid('mixed_13')

        
        class_ranges, labels_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        print(f"On a massively reduced test set with {len(labels_map)} different classes, less complex, representative of common object")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "big":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(400,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "medium":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(100,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "small":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(25,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    print("\n-------------------------")

    print("Some example of classes:")
    print(imagenet_classes[:10])

    print("\n")

    text_zeroshot_weight = compute_text_weight_zeroshot(model=model,tokenizer=tokenizer,device=device,classnames=imagenet_classes,template=imagenet_prompt)

    counter = 0
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(test_loader):
            
          
            
            #images = feature_extractor(images,return_tensors="pt")

            #images = images["pixel_values"]
            
            images = images.to(device)
            target = target.to(device)
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_zeroshot_weight

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}\n")

    return top1,top5






def compute_text_weight_zeroshot(model,tokenizer,device,classnames, template):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = template.format(classname)  #format with class

        
            texts_encoded = tokenizer(texts,padding="max_length",max_length=CFG.max_length) #tokenize

            batch_text = {"input_ids": torch.as_tensor(texts_encoded["input_ids"]).unsqueeze(0).to(device), "attention_mask": torch.as_tensor(texts_encoded["attention_mask"]).unsqueeze(0).to(device)}
            
            class_embeddings = model.encode_text(batch_text) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights



#########################  FLICKR I2T and T2I RETRIEVAL ########################


def i2t_t2i_retrieval(model,dataset,tokenizer,feature_extractor,device):


    test_loader = get_dataloader(dataset=dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,rank=0,world_size=1,batch_size=CFG.test_batch_size,shuffle=False,num_workers=CFG.num_workers,split="test")

    with torch.no_grad():

        top1_i2t, top5_i2t,top10_i2t,top1_t2i,top5_t2i,top10_t2i, n = 0., 0., 0. ,0. ,0. ,0. ,0.
        for batch in tqdm(test_loader):

            image = batch["image"].to(device)
            text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}

            #print(text["input_ids"].shape)
            #print(text["attention_mask"].shape)
            # compute image_features
            image_features = model.encode_image(image)
            # For CLIP only
            #image_features = model.get_image_features(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # compute text features
            text_features = model.encode_text(text)
            # For CLIP only
            #text_features = model.get_text_features(**text)
            
            
            text_features /= text_features.norm(dim=-1, keepdim=True)

            

            sim_i2t = 100. * image_features @ text_features.T

            target = torch.eye(sim_i2t.shape[0]).to(device)
            
            target = target.argmax(dim=0)


            # measure accuracy
            acc1, acc5, acc10 = accuracy(sim_i2t, target, topk=(1, 5, 10))
            top1_i2t += acc1
            top5_i2t += acc5
            top10_i2t += acc10

            acc1, acc5, acc10 = accuracy(sim_i2t.T, target, topk=(1, 5, 10))
            top1_t2i += acc1
            top5_t2i += acc5
            top10_t2i += acc10

            n += image.size(0)

    top1_i2t = (top1_i2t / n) * 100
    top5_i2t = (top5_i2t / n) * 100 
    top10_i2t = (top10_i2t / n) * 100
    


    print(f"Image 2 Text Retrieval on {dataset}:")
    print(f"Top-1 accuracy: {top1_i2t:.2f}")
    print(f"Top-5 accuracy: {top5_i2t:.2f}")
    print(f"Top-10 accuracy: {top10_i2t:.2f}")

    top1_t2i = (top1_t2i / n) * 100
    top5_t2i = (top5_t2i / n) * 100 
    top10_t2i = (top10_t2i / n) * 100 

    print(f"Text 2 Image Retrieval on {dataset}:")
    print(f"Top-1 accuracy: {top1_t2i:.2f}")
    print(f"Top-5 accuracy: {top5_t2i:.2f}")
    print(f"Top-10 accuracy: {top10_t2i:.2f}")

    return top1_i2t,top5_i2t,top10_i2t,top1_t2i,top5_t2i,top10_t2i


############################ ACCURACY #######################

def accuracy(output, target, topk=(1,)):
    indices = output.topk(max(topk), 1, True, True)[1].t()
    correct = indices.eq(target.expand_as(indices))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



############################ MAIN #######################

if __name__ == "__main__":

  

    test()
