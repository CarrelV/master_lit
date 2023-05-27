import sys, os,copy
import collections
import torch
import torch_pruning as tp
import transformers
import config as CFG
from models_CLIP import CLIPMoco
import numpy as np
from transformers import logging

from tokenizer import get_tokenizer,get_feature_extractor
from pruning import pruning_BERT_without_residual

from fisher import compute_fisher
from dataloader import get_local_dataloader

from utils_models import modify_model_after_init

def resume_model(model):
    
    print(f"Config: {CFG.configuration}, text model size: {CFG.text_model_size}, image model size: {CFG.image_model_size}")

    trainable_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.text_encoder.parameters())

    print(f"For the Text encoder:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.text_projection.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.text_projection.parameters())

    print(f"For the Text head:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_encoder.parameters())

    print(f"For the Image encoder:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.image_projection.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_projection.parameters())

    print(f"For the Image head:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

if __name__ == "__main__":
    
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("start")
    model = CLIPMoco().to(device)
    

    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    #print(feature_extractor)
    text = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)

    dummy_image = np.zeros((256,256,3), np.uint8)
    image = feature_extractor(dummy_image,return_tensors="pt").to(device)
    print("Hallo")

    print("\n params which will be finetuned:")
    #for n, p in model.image_encoder.named_parameters():

        
        #print(n)
    
    if CFG.side_text_weights_copy or CFG.side_image_weights_copy:
       
        importance_measure = compute_fisher(model, get_local_dataloader(dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=1,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)
        print("importance measure computed")
        model = modify_model_after_init(model,tokenizer,feature_extractor,importance_measure)

    resume_model(model)
    #print(model)

    '''print("\n params which will be finetuned:")
    for n, p in model.named_parameters():

        if p.requires_grad:
            print(n)'''

    #Parameter
    '''params = []
    if CFG.text_backbone_finetune:
        params.append({"params" : model.text_encoder.side_encoder.parameters(), "lr" : 1e-5})
    if CFG.image_backbone_finetune:
        params.append({"params" : [p for n,p in model.image_encoder.named_parameters()if "side_encoder" in n], "lr" : CFG.image_encoder_lr})

        params.append({"params" : [p for n,p in model.image_encoder.named_parameters()if "side_encoder" not in n], "lr" : CFG.image_encoder_lr})
    params.append({"params" : model.text_projection.parameters(), "lr" : CFG.text_head_lr})
    params.append({"params" : model.image_projection.parameters(), "lr" : CFG.image_head_lr})'''
    
    '''print("Side encoder")
    for n,p in model.image_encoder.named_parameters():
        if "side_encoder" in n:
            print(n)

    print("Not side encoder")
    for n,p in model.image_encoder.named_parameters():
        if "side_encoder" not in n and p.requires_grad:
            print(n)'''
    #for para in model.text_encoder.parameters():
    #    print(para)
    #print("Finish")

    #outputs = model(image["pixel_values"],text)

    #print(outputs)
    #print(outputs)
#  

