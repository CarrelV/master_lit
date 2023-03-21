import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging
import torchvision.transforms as transforms
#from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

from model_test import CLIPMoco
#from models import CLIPMoco

import config as CFG
from tokenizer import get_tokenizer,get_feature_extractor
from dataloader import get_dataloader
from utils import read_imagenet_class

def test():

    print("Start testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()

    
    feature_extractor = get_feature_extractor(CFG.vision_model_name)
    tokenizer = get_tokenizer(CFG.text_model_name)


    model = CLIPMoco().to(device)


    # Load the weights for the backbone
    if CFG.text_backbone_finetune:
        model.text_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_enc_best_{CFG.weight_version}.pt"))
    if CFG.image_backbone_finetune:
        model.image_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_enc_best_{CFG.weight_version}.pt"))
    # Load the heads
    model.text_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_proj_best_{CFG.weight_version}.pt"))
    model.image_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_proj_best_{CFG.weight_version}.pt"))
    '''
    #testing with a pretrained clip
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    feature_extractor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    '''
    print("Model loaded")
    

    model.eval()

    print("-------------------------")
    print(f"For the model {CFG.configuration_to_test}")
    print("-------------------------")
    #imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)

    flickr_retrieval(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)

######################## IMAGENET 0 SHOT ###############



def imagenet_0shot(model,tokenizer,feature_extractor,device):

    transform_ImageNet = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385), 
                                 (0.285, 0.277, 0.286)),
            ])

    ds = ImageNetV2Dataset(transform=transform_ImageNet)
    test_loader = DataLoader(ds, batch_size=CFG.test_batch_size, num_workers=CFG.num_workers)

    imagenet_prompt ='a photo of a {}.'

    imagenet_classes = read_imagenet_class()

    text_zeroshot_weight = compute_text_weight_zeroshot(model=model,tokenizer=tokenizer,device=device,classnames=imagenet_classes,template=imagenet_prompt)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(test_loader):
         
            images = feature_extractor(images.squeeze(0),return_tensors="pt")
            
            images = images["pixel_values"]
            
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

    print("0 Shot classification on ImageNetV2:")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")



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


def flickr_retrieval(model,tokenizer,feature_extractor,device):


    test_loader = get_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,rank=0,world_size=1,batch_size=CFG.batch_size,shuffle=False,num_workers=CFG.num_workers,split="test")

    with torch.no_grad():

        top1_i2t, top5_i2t,top1_t2i,top5_t2i, n = 0., 0., 0. ,0. ,0.
        for batch in test_loader:

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
            acc1, acc5 = accuracy(sim_i2t, target, topk=(1, 5))
            top1_i2t += acc1
            top5_i2t += acc5

            acc1, acc5 = accuracy(sim_i2t.T, target, topk=(1, 5))
            top1_t2i += acc1
            top5_t2i += acc5

            n += image.size(0)

    top1_i2t = (top1_i2t / n) * 100
    top5_i2t = (top5_i2t / n) * 100 

    


    print("Image 2 Text Retrieval on Flickr30k:")
    print(f"Top-1 accuracy: {top1_i2t:.2f}")
    print(f"Top-5 accuracy: {top5_i2t:.2f}")

    top1_t2i = (top1_t2i / n) * 100
    top5_t2i = (top5_t2i / n) * 100 

    print("Text 2 Image Retrieval on Flickr30k:")
    print(f"Top-1 accuracy: {top1_t2i:.2f}")
    print(f"Top-5 accuracy: {top5_t2i:.2f}")


############################ ACCURACY #######################

def accuracy(output, target, topk=(1,)):
    indices = output.topk(max(topk), 1, True, True)[1].t()
    correct = indices.eq(target.expand_as(indices))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]




if __name__ == "__main__":

  

    test()
