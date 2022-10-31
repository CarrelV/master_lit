from tqdm import tqdm
import config as CFG
import wandb
import torch

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg_loss,self.sum_loss, self.count = [0] * 3

    def update(self, loss, count=1):
        self.count += count
        self.sum_loss += loss * count
        self.avg_loss = self.sum_loss / self.count
        

    def __repr__(self):
        text = f"{self.name}: avg_loss = {self.avg_loss:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def train_one_epoch(model, loss_fn, train_loader, optimizer,device):
    
    loss_meter = AvgMeter()

    tqdm_object = tqdm(train_loader, total=len(train_loader))
   
    
    for batch in tqdm_object:

        image = batch["image"].to(device)
        text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        
        # Generate key for this batch, and update the queue outside of the forward pass
        with torch.no_grad():
           
            # Update the momentum encoder
            model.module._momentum_update_key_encoders()

            # Compute the keys
            key_image_features = model.module.key_encode_image(image).to(device)
            key_text_features = model.module.key_encode_text(text).to(device)

            # Get the queue 
            key_image_from_queue = model.module.image_queue.clone().detach().to(device)
            key_text_from_queue = model.module.text_queue.clone().detach().to(device)


            # Now the keys are the cat of new and stored queue
            keys_for_this_batch = {"image_embed" : torch.cat([key_image_features, key_image_from_queue], dim=0), "text_embed": torch.cat([key_text_features, key_text_from_queue], dim=0)}
       
        
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        #compute prediction for the batch
        output = model(image,text)
        
        
        #compute loss and its gradients
        loss = loss_fn(output,keys_for_this_batch)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Dequeue and enqueue the new keys
        with torch.no_grad():
            
            model.module._dequeue_and_enqueue(key_image_features,key_text_features)

        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss, count)

        wandb.log({"step Training loss": loss_meter.avg_loss, "step Learning Rate" : get_lr(optimizer)  } )
        tqdm_object.set_postfix(train_loss=loss_meter.avg_loss.item())
        
        
    return loss_meter



################## VALIDATION EPOCH #################################



def valid_one_epoch(model,loss_fn,valid_loader,device):

    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    for batch in tqdm_object:

        image = batch["image"].to(device)
        text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}

        # Generate key for this batch, but don't update queue nor momentum update 
        with torch.no_grad():
            
            # Compute the keys
            key_image_features = model.module.key_encode_image(image).to(device)
            key_text_features = model.module.key_encode_text(text).to(device)

            # Get the queue 
            key_image_from_queue = model.module.image_queue.clone().detach().to(device)
            key_text_from_queue = model.module.text_queue.clone().detach().to(device)

            # Now the keys are the cat of new and stored queue
            keys_for_this_batch = {"image_embed" : torch.cat([key_image_features, key_image_from_queue], dim=0), "text_embed": torch.cat([key_text_features, key_text_from_queue], dim=0)}
            
        #compute prediction for the batch
        output = model(image,text)
        
        #compute loss and its gradients
        loss = loss_fn(output,keys_for_this_batch)

        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss, count)

        wandb.log({"step Validation loss": loss_meter.avg_loss} )
        tqdm_object.set_postfix(valid_loss=loss_meter.avg_loss.item())

    return loss_meter

