import torch
import config as CFG
import torch.nn.functional as F
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * CFG.temperature)
        

    def forward(self, outputs):
        
        image_embeddings = outputs["image_embed"]
        text_embeddings = outputs["text_embed"]
        
        logit_scale = self.logit_scale.exp()

        # normalized features
        image_embeds = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        
        logits_per_text = text_embeds @ image_embeds.t() * logit_scale
        logits_per_image = logits_per_text.t()
        
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        loss = clip_loss(logits_per_image,logits_per_text,labels)
        
        return loss


class CLIPMoCOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * CFG.temperature)
        

    def forward(self, outputs,keys):
        
        image_embeddings = outputs["image_embed"]
        text_embeddings = outputs["text_embed"]
        
        key_image_embeddings = keys["image_embed"]
        key_text_embeddings = keys["text_embed"]

        logit_scale = self.logit_scale.exp()

        # normalized features
        image_embeds = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeds = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        key_image_embeds = key_image_embeddings / key_image_embeddings.norm(dim=-1, keepdim=True)
        key_text_embeds = key_text_embeddings / key_text_embeddings.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        
        logits_per_text = text_embeds @ key_image_embeds.t() * logit_scale
        logits_per_image = image_embeds @ key_text_embeds.t() * logit_scale

        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        loss = clip_loss(logits_per_image,logits_per_text,labels)
        
        return loss
        #return {"loss" : loss, "logits_per_text" : logits_per_text, "logits_per_image": logits_per_image}



def clip_loss(logits_per_image: torch.Tensor,logits_per_text,labels) -> torch.Tensor:
    caption_loss = F.cross_entropy(logits_per_text,labels)
    image_loss = F.cross_entropy(logits_per_image,labels)
    return (caption_loss + image_loss) / 2.0