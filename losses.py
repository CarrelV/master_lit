import torch
import config as CFG
import torch.nn.functional as F
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = CFG.temperature
        

    def forward(self, outputs):
        
        image_embeddings = outputs["image_embed"]
        text_embeddings = outputs["text_embed"]
        logits = outputs["logits"]

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        
        return {"loss" : loss, "loss mean": loss.mean(), "text_loss" : texts_loss.mean(), "image_loss": images_loss.mean()}



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()