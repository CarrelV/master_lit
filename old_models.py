import config as CFG

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import ViTModel, ViTConfig

from copy import deepcopy


###################### TEXT TOWER ####################################

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_model_name, pretrained=CFG.text_backbone_pretrained, trainable=CFG.text_backbone_finetune):
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
    def __init__(self, model_name=CFG.image_model_name, pretrained=CFG.image_backbone_pretrained, trainable=CFG.image_backbone_finetune):
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


###################### PROJECTION HEAD ####################################

# Always used for the image head, and the image baseline head
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        return x


###################### SMALL MLP HEAD ####################################

# Serves as small MLP for LiT version
class SmallMLPHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(2048, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


###################### LARGE MLP HEAD ####################################

# Serves as large MLP for APE paper and also the APE/LiT combination (additional)
class LargeMLPHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(2048, 2048)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(2048, 2048)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.fc5 = nn.Linear(2048, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.layer_norm(x)
        return x



##################### MOCO version for CLIP style Model ####################################
        
# Same as CLIP Projection, but implementing MOCO to be able to finetune both Text and Image tower as well, and keep a lot
# of negative contrastive exemples despite the smaller batch size

class CLIPMoco(nn.Module):
    def __init__(
        self,
        text_head_config = CFG.text_head_config,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
        proj_dim = CFG.projection_dim,
        finetune_image=CFG.image_backbone_finetune,
        finetune_text=CFG.text_backbone_finetune,
        K=CFG.K,
        m=0.9
    ):
        super().__init__()
        self.text_head_config = text_head_config
        
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        if self.text_head_config == "simple_proj":
            self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        elif self.text_head_config == "small_mlp":
            self.text_projection = SmallMLPHead(embedding_dim=text_embedding)
        elif self.text_head_config == "large_mlp":
            self.text_projection = LargeMLPHead(embedding_dim=text_embedding)

        self.proj_dim = proj_dim
        self.temperature = temperature
        self.finetune_image = finetune_image
        self.finetune_text = finetune_text

        # MOCO parameters
        self.K = K
        self.m = m

        # Init key encoders
        self.image_key_encoder = deepcopy(self.image_encoder)
        for param_k in self.image_key_encoder.parameters():param_k.requires_grad = False

        self.text_key_encoder = deepcopy(self.text_encoder)
        for param_k in self.image_key_encoder.parameters(): param_k.requires_grad = False

        self.image_key_projection = deepcopy(self.image_projection)
        for param_k in self.image_key_projection.parameters(): param_k.requires_grad = False

        self.text_key_projection = deepcopy(self.text_projection)
        for param_k in self.text_key_projection.parameters():param_k.requires_grad = False

        # Init Queues
        self.image_queue = torch.randn(self.K,self.proj_dim)
        self.text_queue = torch.randn(self.K,self.proj_dim)

        self.queue_ptr = 0

    def encode_text(self,text):
        if not self.finetune_text:
            with torch.no_grad():
                text_features = self.text_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        
        else:
            text_features = self.text_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])

        # Getting Text Embeddings (output of proj heads)
        text_embeddings = self.text_projection(text_features)

        return  text_embeddings


    @torch.no_grad()
    def key_encode_text(self,text):
        
        text_features = self.text_key_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
            
        # Getting Text Embeddings (output of proj heads)
        text_embeddings = self.text_key_projection(text_features)

        return  text_embeddings

    def encode_image(self,image):
        if not self.finetune_image:
            with torch.no_grad():
                image_features = self.image_encoder(image)

        
        else:
            image_features = self.image_encoder(image)


        # Getting Image Embeddings (output of proj heads)
        image_embeddings = self.image_projection(image_features)


        return image_embeddings


    @torch.no_grad()
    def key_encode_image(self,image):
        
        image_features = self.image_key_encoder(image)

        
        # Getting Image Embeddings (output of proj heads)
        image_embeddings = self.image_key_projection(image_features)


        return image_embeddings

    ## Update all key parameters (both encoders and projection module)
    @torch.no_grad()
    def _momentum_update_key_encoders(self):
        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.text_encoder.parameters(), self.text_key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
        for param_q, param_k in zip(self.image_projection.parameters(), self.image_key_projection.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.text_projection.parameters(), self.text_key_projection.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    # Add new minibatch _k to queue and remove the oldest minibatch in queue
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_k, text_k):

        image_k = concat_all_gather(image_k)
        text_k = concat_all_gather(text_k)

        bs = image_k.size(0)
        assert self.K % bs == 0  # for simplicity
        
        self.image_queue[self.queue_ptr:self.queue_ptr+bs, :] = image_k
        self.text_queue[self.queue_ptr:self.queue_ptr+bs, :] = text_k
        self.queue_ptr = (self.queue_ptr + bs) % self.K  # move pointer to avoid smaller batch at the end 

    # Computing the keys outside of the forward function as discussed on 27.11
    def forward(self, image,text):
        
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(text)
        
        return {"image_embed": image_embeddings, "text_embed": text_embeddings}

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

