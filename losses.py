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
        

        # normalized features
        image_embeds = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = clip_loss(logits_per_text)
        
        return {"loss" : loss, "logits_per_text" : logits_per_text, "logits_per_image": logits_per_image}



def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0