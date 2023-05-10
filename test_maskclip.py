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

from utils_maskclip import get_text_weights,get_image_weights
from dataloader import get_dataloader

from model_maskCLIP import CLIPMask

if __name__ == "__main__":
    
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Prepare original model to get weights")
    model = CLIPMoco().to(device)


    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    model.text_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_enc_best_{CFG.weight_version}.pt",map_location=device))
    model.image_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_enc_best_{CFG.weight_version}.pt",map_location=device))
    model.text_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_text_proj_best_{CFG.weight_version}.pt",map_location=device))
    model.image_projection.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_proj_best_{CFG.weight_version}.pt",map_location=device))
    

    text_weight,text_categories = get_text_weights(model,tokenizer,["ade"])
    #print(feature_extractor)

    print(f"text_weight shape: {text_weight.shape}")


    image_weights_1, image_weights_2 = get_image_weights(model)

    print(f"image_weights_1 shape: {image_weights_1.shape}")
    print(f"image_weights_2 shape: {image_weights_2.shape}")

    print("Prepare mask clip model")
    model = CLIPMask(text_categories=text_categories,text_channels=CFG.text_embedding)

    model.image_encoder.load_state_dict(torch.load(f"weights/{CFG.configuration_to_test}_img_enc_best_{CFG.weight_version}.pt",map_location=device))
    
    
    print("Loading the weights")
    model.segmentation_head.load_text_embeddings(text_weight)
    model.segmentation_head.load_visual_projs(image_weights_1, image_weights_2)

    print("Mask clip model ready")

    test_loader = get_dataloader(dataset="ade",tokenizer=tokenizer,feature_extractor=feature_extractor,rank=0,world_size=1,batch_size=1,shuffle=False,num_workers=CFG.num_workers,split="val")
    
    counter = 0
    with torch.no_grad():

        for batch in test_loader:
            image = batch["image"].to(device)
            seg_map = batch["seg_map"]
            
            output = model(image)
            

            print("Original segmentation map:")
            print(f"shape: {seg_map.shape}")

            print(seg_map)

            print("Predicted segmap:")
            print(f"sahpe: {output.shape}")
            print(output)

            

