from tqdm import tqdm
import config as CFG
import torch
import numpy as np

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

data_to_return = []

def train_one_epoch(model, loss_fn, train_loader, optimizer,lr_scheduler,device):
    
    loss_meter = AvgMeter()
    # For cc3m, using local dataloader with hardcoded fixes for now
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
    #for batch in train_loader:

        a = torch.cuda.memory_allocated(device)


        image = batch["image"].to(device)
        text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        
        # Generate key for this batch, and update the queue outside of the forward pass
        with torch.no_grad():
           
            # Update the momentum encoder
            model._momentum_update_key_encoders()

            # Compute the keys
            key_image_features = model.key_encode_image(image).to(device)
            key_text_features = model.key_encode_text(text).to(device)

            # Get the queue 
            key_image_from_queue = model.image_queue.clone().detach().to(device)
            key_text_from_queue = model.text_queue.clone().detach().to(device)


            # Now the keys are the cat of new and stored queue
            keys_for_this_batch = {"image_embed" : torch.cat([key_image_features, key_image_from_queue], dim=0), "text_embed": torch.cat([key_text_features, key_text_from_queue], dim=0)}
       
        
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        #compute prediction for the batch
        output = model(image,text)
        
        b = torch.cuda.memory_allocated(device)

        data_to_return.append(b)
        #print(f"2 - After forward pass: {b}")
        #print(f"2 - Memory consumed by forward pass: {b-a}")
        #compute loss and its gradients
        loss = loss_fn(output,keys_for_this_batch)

        c = torch.cuda.memory_allocated(device)
        data_to_return.append(c)
        #print(f"3 - After computing loss and its gradient pass: {c}")
        loss.backward()

        d = torch.cuda.memory_allocated(device)
        data_to_return.append(d)
        #print(f"4 - After loss backward pass: {d}")
        # Adjust learning weights
        optimizer.step()

        e = torch.cuda.memory_allocated(device)
        data_to_return.append(e)
        #print(f"5 - After optimizer pass: {e}")
        lr_scheduler.step()  # Update learning rate
        
        # Dequeue and enqueue the new keys
        with torch.no_grad():
            
            model._dequeue_and_enqueue(key_image_features,key_text_features)

        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss, count)

        #tqdm_object.set_postfix(train_loss=loss_meter.avg_loss.item())

        break
        
        
    return loss_meter,np.max(data_to_return)



################## VALIDATION EPOCH #################################



def valid_one_epoch(model,loss_fn,valid_loader,device):

    loss_meter = AvgMeter()

    #tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    counter = 0

    #for batch in tqdm_object:
    for batch in valid_loader:
        counter += 1
        if counter >= 140:
            break
        print(f"Val Minibatch: {counter}", end="\r", flush=True)
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

        #tqdm_object.set_postfix(valid_loss=loss_meter.avg_loss.item())

    return loss_meter

