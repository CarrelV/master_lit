import sys, os,copy
import collections
import torch
import torch_pruning as tp
import transformers
import config as CFG

from tokenizer import get_tokenizer,get_feature_extractor
from pruning import pruning_BERT_without_residual

from model_BERTLsT import *

from transformers import AutoTokenizer, BartModel,BartConfig

if __name__ == "__main__":
    
    #print("BART")
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    #model = BartModel.from_pretrained("facebook/bart-base")
    #print(model.state_dict().keys())

    #bart_config = BartConfig.from_pretrained("facebook/bart-base")
    #print(bart_config)
    ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    reduction_factors = [8]
    
    text_config = BertConfig.from_pretrained(CFG.text_model_name)
    model = BertLSTModel(config=text_config)

    #print(text_config)
    tokenizer = get_tokenizer(CFG.text_model_name)

    print("Original model")
    #for n, p in model.named_parameters():

        
        #print(n)
    print(model)

#    print(model.state_dict().keys())
'''
    print("Reduced model")
    for reduction_factor in reduction_factors:
        
        # pruned_state_dict = pruning_v1(model, reduction_factor)

        pruned_state_dict = pruning_BERT_without_residual(model, tokenizer, reduction_factor, importance_measure=None)

 #       print(pruned_state_dict.keys())
        #for n, p in model.named_parameters():

            #if n not in pruned_state_dict:
                #print(n)

        side_config = copy.deepcopy(text_config)

        side_config.intermediate_size = side_config.intermediate_size // CFG.reduction_factor
        side_config.hidden_size = side_config.hidden_size // CFG.reduction_factor

        new_model = BertLSTModel(config=side_config)
        print(new_model)'''