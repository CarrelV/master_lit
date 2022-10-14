from tqdm import tqdm
import config as CFG
import wandb


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
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        #compute prediction for the batch
        output = model(batch)
        
        #compute loss and its gradients
        loss = loss_fn(output)
        loss.backward()

        # Adjust learning weights
        optimizer.step()


        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss, count)

        wandb.log({"loss": loss_meter.avg_loss, "lr" : get_lr(optimizer)  } )
        tqdm_object.set_postfix(train_loss=loss_meter.avg_loss.item())
        
        
    return loss_meter

def valid_one_epoch(model,loss_fn,valid_loader,device):

    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    for batch in tqdm_object:

        batch = {k: v.to(device) for k, v in batch.items()}

        #compute prediction for the batch
        output = model(batch)
        
        #compute loss and its gradients
        loss = loss_fn(output)

        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss, count)

        wandb.log({"valid loss": loss_meter.avg_loss} )
        tqdm_object.set_postfix(valid_loss=loss_meter.avg_loss.item())

    return loss_meter
