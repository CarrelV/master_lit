import functools
import numpy as np
import torch_pruning.dependency as tpd
import torch_pruning.strategy as tps
import sys,os,copy
import torch
from PIL import Image


def select_weights(weights, idxs):
    num_features = len(weights)
    keep_idxs = list(set(range(num_features)) - set(idxs))

    return weights[keep_idxs]

def get_saved_idx(weights,idxs):
    num_features = len(weights)
    keep_idxs = list(set(range(num_features)) - set(idxs))

    return keep_idxs

def pruning_BERT_without_residual(model, tokenizer, reduction_factor, importance_measure=None):
    inputs_dict = tokenizer(["I like dogs.", "I really like your cats."], padding=True, return_tensors="pt")
    # padding="max_length",max_length=CFG.max_length
    inputs_dict = {"input_ids": inputs_dict["input_ids"], "attention_mask": inputs_dict["attention_mask"]} # transform to dict
    
    # Build dependency graph
    DG = tpd.DependencyGraph()
    DG.build_dependency(model, example_inputs = inputs_dict, pruning_dim = -1) 
    # Note to set pruning_dim to -1 to prune BertModel on hidden_states.

    # get a pruning plan by pruning from word embedding

    print("ici")
    strategy = tps.L1Strategy()
    

    
    prune_vals = [1 - 1 / reduction_factor]

    state_dict = model.state_dict()
    
    if importance_measure is None:
        importance_measure = copy.deepcopy(state_dict)

    # construct ordered layers to process
    ordered_target_layers = []

    pruning_idxs_history =  {}

    sub_model = "model.encoder"
    ordered_target_layers.append("model.embeddings.position_embeddings.weight")
    ordered_target_layers.append("model.embeddings.LayerNorm.weight")
    for i in range(model.config.num_hidden_layers):
        ordered_target_layers.append(
            [f"{sub_model}.layer.{i}.attention.self.{n}.weight" for n in ["query", "key", "value"]]
            )
        
        for n in ["query", "key", "value"]:
            ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.self.{n}.bias")

        ordered_target_layers.append([f"{sub_model}.layer.{i}.attention.output.dense.weight"])
        ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.output.dense.bias")

        ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.output.LayerNorm.weight")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.output.LayerNorm.bias")

        ordered_target_layers.append([f"{sub_model}.layer.{i}.intermediate.dense.weight"])
        ordered_target_layers.append(f"{sub_model}.layer.{i}.intermediate.dense.bias")
        ordered_target_layers.append([f"{sub_model}.layer.{i}.output.dense.weight"])
        ordered_target_layers.append(f"{sub_model}.layer.{i}.output.dense.bias")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.output.LayerNorm.weight")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.output.LayerNorm.bias")


    
    pruning_idxs_first_layer = None

    print("là")
    for prune_val in prune_vals:
        new_state_dict = {}
        for layer in ordered_target_layers:
            
            if isinstance(layer, list):

                weights = [state_dict[sub_layer] for sub_layer in layer]
                importances = [importance_measure["text_encoder."+sub_layer] for sub_layer in layer]
                
                # prune according to previous idx
                weights = [select_weights(w.T, pruning_idxs).T for w in weights]
                importances = [select_weights(imp.T, pruning_idxs).T for imp in importances]

                # use the sum of log values instead of the product of values
                prod_imp = 0
                for imp in importances:
                    prod_imp += torch.log(imp)

                pruning_idxs = strategy(weights=prod_imp, amount=prune_val)

                weights = [select_weights(w, pruning_idxs) for w in weights]
                importances = [select_weights(imp, pruning_idxs) for imp in importances]                    


                for l, w in zip(layer, weights):
                    new_state_dict[l] = w

                # update importance measure
                for l, imp in zip(layer, importances):
                    importance_measure["text_encoder."+l] = imp

            elif "position_embeddings.weight" in layer:
                # the first layer
                # will only select next prune idx
                # and direct copy the same weights

                importance = importance_measure["text_encoder."+layer]
                weights = state_dict[layer]
                pruning_idxs = strategy(weights=importance.T, amount=prune_val)


                weights = select_weights(weights.T, pruning_idxs).T
                importance = select_weights(importance.T, pruning_idxs).T


                new_state_dict[layer] = weights
                importance_measure["text_encoder."+layer] = importance
            
            else:
                # layerNorm layer.
                # will only prune the weights according to previous prune idx
                weights = state_dict[layer]
                
                keep_idxs = get_saved_idx(weights,pruning_idxs)

                weights = select_weights(weights, pruning_idxs)

                importance = importance_measure["text_encoder."+layer]
                importance = select_weights(importance, pruning_idxs)
                
                new_state_dict[layer] = weights
                importance_measure["text_encoder."+layer] = importance

                if ("attention" not in layer) and ("bias" not in layer) and ("embeddings" not in layer):
                    pruning_idxs_history[layer] = keep_idxs

            
        state_dict = new_state_dict

    return new_state_dict,pruning_idxs_history


def pruning_ViT_without_residual(model, feature_extractor, reduction_factor, importance_measure=None):
    
    dummy_image = np.zeros((256,256,3), np.uint8)
    image = feature_extractor(dummy_image,return_tensors="pt")
    
    inputs_dict = {"pixel_values": image["pixel_values"]} # transform to dict
    
    # Build dependency graph
    DG = tpd.DependencyGraph()
    DG.build_dependency(model, example_inputs = image["pixel_values"], pruning_dim = -1) 
    # Note to set pruning_dim to -1 to prune ViTModel on hidden_states.

    # get a pruning plan by pruning from word embedding

    
    strategy = tps.L1Strategy()
    

    
    prune_vals = [1 - 1 / reduction_factor]
    

    state_dict = model.state_dict()
    
    if importance_measure is None:
        importance_measure = copy.deepcopy(state_dict)

    # construct ordered layers to process
    ordered_target_layers = []

    sub_model = "model.encoder"
    ordered_target_layers.append("model.embeddings.position_embeddings")
    #ordered_target_layers.append(["embeddings.patch_embeddings.projection.weight"])
    #ordered_target_layers.append(["embeddings.patch_embeddings.projection.bias"])
    for i in range(model.config.num_hidden_layers):
        ordered_target_layers.append(
            [f"{sub_model}.layer.{i}.attention.attention.{n}.weight" for n in ["query", "key", "value"]]
            )
        for n in ["query", "key", "value"]:
            ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.attention.{n}.bias")

        ordered_target_layers.append([f"{sub_model}.layer.{i}.attention.output.dense.weight"])
        ordered_target_layers.append(f"{sub_model}.layer.{i}.attention.output.dense.bias")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.layernorm_before.weight")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.layernorm_before.bias")

        ordered_target_layers.append([f"{sub_model}.layer.{i}.intermediate.dense.weight"])#
        ordered_target_layers.append(f"{sub_model}.layer.{i}.intermediate.dense.bias")
        ordered_target_layers.append([f"{sub_model}.layer.{i}.output.dense.weight"])#
        ordered_target_layers.append(f"{sub_model}.layer.{i}.output.dense.bias")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.layernorm_after.weight")
        ordered_target_layers.append(f"{sub_model}.layer.{i}.layernorm_after.bias")

    
    pruning_idxs_first_layer = None

    pruning_idxs_history = {}

    for prune_val in prune_vals:
        new_state_dict = {}
        for layer in ordered_target_layers:

            if isinstance(layer, list):
               
                weights = [state_dict[sub_layer] for sub_layer in layer]
                importances = [importance_measure["image_encoder."+sub_layer] for sub_layer in layer]

                # prune according to previous idx
                weights = [select_weights(w.T, pruning_idxs).T for w in weights]
                importances = [select_weights(imp.T, pruning_idxs).T for imp in importances]

                # use the sum of log values instead of the product of values
                prod_imp = 0
                for imp in importances:
                    prod_imp += torch.log(imp)

                pruning_idxs = strategy(weights=prod_imp, amount=prune_val)

                weights = [select_weights(w, pruning_idxs) for w in weights]
                importances = [select_weights(imp, pruning_idxs) for imp in importances]

                for l, w in zip(layer, weights):
                    new_state_dict[l] = w

                # update importance measure
                for l, imp in zip(layer, importances):
                    importance_measure["image_encoder."+l] = imp

            elif "position_embeddings" in layer:
                # the first layer
                # will only select next prune idx
                # and direct copy the same weights
                importance = importance_measure["image_encoder."+layer]
                weights = state_dict[layer]
                pruning_idxs = strategy(weights=importance.T, amount=prune_val)

                weights = select_weights(weights.T, pruning_idxs).T
                importance = select_weights(importance.T, pruning_idxs).T

                new_state_dict[layer] = weights
                importance_measure["image_encoder."+layer] = importance

            
            else:
                # layerNorm or bias
                weights = state_dict[layer]
                
                keep_idxs = get_saved_idx(weights,pruning_idxs)

                weights = select_weights(weights, pruning_idxs)

                importance = importance_measure["image_encoder."+layer]
                importance = select_weights(importance.T, pruning_idxs).T
                
                new_state_dict[layer] = weights
                importance_measure["image_encoder."+layer] = importance

                if ("layernorm_after" in layer) and ("bias" not in layer):
                    pruning_idxs_history[layer] = keep_idxs

        state_dict = new_state_dict

    return new_state_dict,pruning_idxs_history