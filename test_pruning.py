import sys, os,copy
import collections
import torch
import torch_pruning as tp
import transformers
import config as CFG
from models_CLIP import CLIPMoco

from tokenizer import get_tokenizer,get_feature_extractor
from pruning import pruning_BERT_without_residual,pruning_ViT_without_residual

from model_BERTLsT import *
from model_ViTLsT import *
from transformers import AutoTokenizer, BartModel,BartConfig
from fisher import compute_fisher
from dataloader import get_local_dataloader

if __name__ == "__main__":
    
    #print("BART")
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    #model = BartModel.from_pretrained("facebook/bart-base")
    #print(model.state_dict().keys())

    #bart_config = BartConfig.from_pretrained("facebook/bart-base")
    #print(bart_config)
    ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()


    reduction_factors = [8]
    
    model = CLIPMoco()

    #print(text_config)
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    #print("Original model")
    #for n, p in model.named_parameters():

        
    #    print(n)


    importance_measure = compute_fisher(model, get_local_dataloader(dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=8,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)

    print("\n For the Image \n \n")
  

    print("Reduced model")
    for reduction_factor in reduction_factors:
        
        # pruned_state_dict = pruning_v1(model, reduction_factor)
        pruned_state_dict_text = pruning_BERT_without_residual(model.text_encoder,tokenizer,reduction_factor,importance_measure=importance_measure)
        pruned_state_dict = pruning_ViT_without_residual(model.image_encoder, feature_extractor, reduction_factor, importance_measure=importance_measure)
        
        print("pruned state \n \n")

        print(pruned_state_dict.keys())

        print("Not pruned")
        for n, p in model.image_encoder.named_parameters():

            if n not in pruned_state_dict:
                print(n)

        '''
        side_config = copy.deepcopy(text_config)

        side_config.intermediate_size = side_config.intermediate_size // CFG.reduction_factor
        side_config.hidden_size = side_config.hidden_size // CFG.reduction_factor

        new_model = BertLSTModel(config=side_config)
        print(new_model)'''