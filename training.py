from tqdm import tqdm
import config as CFG
import wandb


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg_loss,self.sum_loss,self.avg_text_l,self.sum_text_l,self.avg_img_l, self.sum_img_l, self.count = [0] * 7

    def update(self, loss,text_l,img_l, count=1):
        self.count += count
        self.sum_loss += loss * count
        self.avg_loss = self.sum_loss / self.count
        self.sum_text_l += text_l * count
        self.avg_text_l = self.sum_text_l / self.count
        self.sum_img_l += img_l * count
        self.avg_img_l = self.sum_img_l / self.count

    def __repr__(self):
        text = f"{self.name}: avg_loss = {self.avg_loss:.4f}, avg_text_loss = {self.avg_text_l:.4f}, avg_img_loss = {self.avg_img_l:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

        
def train_one_epoch(model, loss_fn, train_loader, optimizer):
    
    loss_meter = AvgMeter()

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
        
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        #compute prediction for the batch
        output = model(batch)
        
        #compute loss and its gradients
        loss = loss_fn(output)
        loss["loss mean"].backward()

        # Adjust learning weights
        optimizer.step()


        # Gather data and report
        count = batch["image"].size(0)
        loss_meter.update(loss["loss mean"],loss["text_loss"],loss["image_loss"], count)

        wandb.log({"loss": loss_meter.avg_loss,"text loss": loss_meter.avg_text_l,"image loss": loss_meter.avg_img_l, "lr" : get_lr(optimizer)  } )
        tqdm_object.set_postfix(train_loss=loss_meter.avg_loss.item())
        
        
    return loss_meter