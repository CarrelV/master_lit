import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging

from models import CLIPModel
import config as CFG
from tokenizer import get_tokenizer,get_feature_extractor
from dataloader import transform_ImageNet,get_dataloader
from utils import read_imagenet_class

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()

    
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    tokenizer = get_tokenizer(CFG.text_model_name)

    model = CLIPModel().to(device)

    checkpoint_image = torch.load(CFG.image_checkpoint)
    checkpoint_text = torch.load(CFG.text_checkpoint)

    model.image_projection.load_state_dict(checkpoint_image)
    model.text_projection.load_state_dict(checkpoint_text)


    imagenet_0shot(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)

    flickr_retrieval(model=model,tokenizer=tokenizer,feature_extractor=feature_extractor,device=device)

######################## IMAGENET 0 SHOT ###############


def imagenet_0shot(model,tokenizer,feature_extractor,device):

    ds = ImageNetV2Dataset(transform=transform_ImageNet)
    test_loader = DataLoader(ds, batch_size=CFG.test_batch_size, num_workers=CFG.num_workers)

    imagenet_prompt ='a photo of a {}.'

    imagenet_classes = read_imagenet_class()

    text_zeroshot_weight = compute_text_weight_zeroshot(model=model,tokenizer=tokenizer,device=device,classnames=imagenet_classes,template=imagenet_prompt)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)
            
            # predict
            image_encoded = feature_extractor(images.squeeze(0).cpu(),"pt").to(device)

            image_features = model.encode_image(image_encoded["pixel_values"])
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

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

#########################  FLICKR I2T and T2I RETRIEVAL ########################

def i2t(sim: torch.Tensor):
    
    order = sim.argsort()
    
    top1 = 0
    top5 = 0
    top10 = 0

    for i in range(len(order[0])):
        if i == order[i,0]:
            top1 += 1
        if i in order[i,0:5]:
            top5 += 1
        if i in order[i,:10]:
            top10 += 1

    return 100 * top1 / len(order[0]), 100 * top5 / len(order[0]) , 100 * top10 / len(order[0])

def t2i(sim: torch.Tensor):
    
    sim = sim.T
    order = sim.argsort()
    
    top1 = 0
    top5 = 0
    top10 = 0

    for i in range(len(order[0])):
        if i == order[i,0]:
            top1 += 1
        if i in order[i,0:5]:
            top5 += 1
        if i in order[i,:10]:
            top10 += 1

    return 100 * top1 / len(order[0]), 100 * top5 / len(order[0]) ,100 * top10 / len(order[0])


def flickr_retrieval(model,tokenizer,feature_extractor,device):


    train_loader = get_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=False,num_workers=CFG.num_workers,split="test")

    with torch.no_grad():

        for batch in train_loader:

            image = batch["image"].to(device)
            text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}

            # compute image_features
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # compute text features
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            

            similarities = image_features @ text_features.T

            i2t_r1,i2t_r5,i2t_r10 = i2t(similarities)
            t2i_r1,t2i_r5,t2i_r10 = t2i(similarities)

            print("Image 2 Text Retrieval on Flickr30k:")
            print(f"Top-1 accuracy: {i2t_r1:.2f}")
            print(f"Top-5 accuracy: {i2t_r5:.2f}")
            print(f"Top-10 accuracy: {i2t_r10:.2f}")

            print("Text 2 Image Retrieval on Flickr30k:")
            print(f"Top-1 accuracy: {t2i_r1:.2f}")
            print(f"Top-5 accuracy: {t2i_r5:.2f}")
            print(f"Top-10 accuracy: {t2i_r10:.2f}")












if __name__ == "__main__":

  

    test()