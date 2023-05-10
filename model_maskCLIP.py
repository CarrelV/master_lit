import config as CFG

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_BERTLsT import BertModel,BertConfig,BertLSTModel
from model_ViTLsT import ViTModel,ViTConfig, ViTLSTModel


###################### MASK CLIP HEAD ####################################

class MaskClipHead(nn.Module):
    def __init__(self,text_categories,text_channels,ks_thresh=0.,
                    pd_thresh=0.) -> None:
        super().__init__()

        self.in_channels = CFG.image_embedding
        self.text_categories = text_categories
        self.text_channels = text_channels

        self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_channels),requires_grad=False)
        nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        
        
        self.proj_1 = nn.Conv2d(self.in_channels, 1024, 1, bias=False)
        self.proj_2 = nn.Conv2d(1024, text_channels, 1, bias=False)


        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh

        for param in self.proj_1.parameters():
            param.requires_grad = False
        for param in self.proj_2.parameters():
            param.requires_grad = False





    def load_text_embeddings(self,text_emb_weights):
      
        self.text_embeddings[:, :] = text_emb_weights[:, :]
        print(f'Loaded text embeddings weights into the text_embeddings MaskCLIPHead')

    def load_visual_projs(self,visual_proj_weights_1,visual_proj_weights_2):
        

        self.proj_1.weight.copy_(visual_proj_weights_1[:, :,None,None]) 

        self.proj_2.weight.copy_(visual_proj_weights_2[:, :,None,None]) 
        print(f'Loaded image proj weights into the proj MaskCLIPHead')



    
    def forward(self, inputs):
        x = inputs[-1]
        q, k, v, cls_token = None, None, None, None
        
        if isinstance(x, list) and len(x) == 4:
            x, q, k, v = x
        if isinstance(x, list) and len(x) == 2:
            x, cls_token = x
        if v is not None:
            feat = self.proj_1(v)
            feat = self.proj_2(feat)
        else:
            feat = self.proj_1(x)
            feat = self.proj_2(feat)
        if cls_token is not None:
            cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        
        output = self.cls_seg(feat)
        if not self.training:
            output = self.refine_output(output, k)

        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        
        return output

    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        raise RuntimeError('MaskClip is not trainable. Try MaskClip+ instead.')

###################### IMAGE TOWER ####################################


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.image_model_name, trainable=False):
        super().__init__()
        self.config = ViTConfig.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
             
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, image):
        
        output = self.model(image)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state

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

###################### CLIP Mask ####################################

class CLIPMask(nn.Module):
    def __init__(
        self,
        text_categories,
        text_channels=CFG.text_embedding,
        image_tower_config = CFG.image_tower_config,
    ):
        super().__init__()
        self.text_categories = text_categories
        self.text_channel = text_channels
        self.image_tower_config = image_tower_config

        if self.image_tower_config == "classic":
            self.image_encoder = ImageEncoder()
        elif self.image_tower_config == "LST":
            self.image_encoder = ViTLSTModel.from_pretrained(CFG.image_model_name)

        self.segmentation_head = MaskClipHead(text_categories,text_channels)

    def encode_image(self,image):

        with torch.no_grad():

            image_encoder_output = self.image_encoder(image)

        return image_encoder_output

    def forward(self,image):

        image_embeddings = self.encode_image(image)
        image_prediction = self.segmentation_head(image_embeddings)

        return image_prediction
