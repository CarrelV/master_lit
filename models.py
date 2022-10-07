import config as CFG

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import ViTModel, ViTConfig


###################### TEXT TOWER ####################################

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = BertModel.from_pretrained(model_name)
        else:

            self.model = BertModel(config=BertConfig.from_pretrained(model_name))
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]




###################### IMAGE TOWER ####################################


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.vision_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            self.model = ViTModel(config=ViTConfig.from_pretrained(model_name))
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        

    def forward(self, image):
        
        return self.model(image)