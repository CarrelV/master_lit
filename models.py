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

        self.target_token_idx = 0

    def forward(self, image):
        
        output = self.model(image)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


###################### PROJECTION HEAD on top ####################################

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


###################### COMPLETE MODEL ####################################


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
        trainable=CFG.trainable
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.trainable = trainable

    def encode_text(self,text):
        if not self.trainable:
            with torch.no_grad():
                text_features = self.text_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        
        else:
            text_features = self.text_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])

        # Getting Image and Text Embeddings (with same dimension) (output of proj heads)
        text_embeddings = self.text_projection(text_features)

        return  text_embeddings

    def encode_image(self,image):
        if not self.trainable:
            with torch.no_grad():
                image_features = self.image_encoder(image)

        
        else:
            image_features = self.image_encoder(image)


        # Getting Image and Text Embeddings (with same dimension) (output of proj heads)
        image_embeddings = self.image_projection(image_features)


        return image_embeddings
    
    def forward(self, batch):
        # Getting Image and Text Features (output of towers)

        if not self.trainable:
            with torch.no_grad():
                image_features = self.image_encoder(batch["image"])
                text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        else:
            image_features = self.image_encoder(batch["image"])
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # Getting Image and Text Embeddings (with same dimension) (output of proj heads)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return {"image_embed": image_embeddings, "text_embed": text_embeddings}
        
        
