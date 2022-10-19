import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CLIPModel
import config as CFG
from tokenizer import get_tokenizer
from dataloader import transform_test
from utils import read_imagenet_class

def test_0shot():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = ImageNetV2Dataset(transform=transform_test)
    test_loader = DataLoader(ds, batch_size=CFG.test_batch_size, num_workers=CFG.num_workers)

    tokenizer = get_tokenizer(CFG.text_model_name)

    model = CLIPModel().to(device)

    checkpoint_image = torch.load(CFG.image_checkpoint)
    checkpoint_text = torch.load(CFG.text_checkpoint)

    model.image_projection.load_state_dict(checkpoint_image)
    model.text_projection.load_state_dict(checkpoint_text)


    

    imagenet_prompt ='a photo of a {}.'

    imagenet_classes = read_imagenet_class()

    text_zeroshot_weight = compute_text_weight_zeroshot(model=model,tokenizer=tokenizer,device=device,classnames=imagenet_classes,template=imagenet_prompt)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_zeroshot_weight

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

def compute_text_weight_zeroshot(model,tokenizer,device,classnames, template):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = template.format(classname)  #format with class

        
            texts_encoded = tokenizer(texts,padding="max_length",max_length=CFG.max_length) #tokenize

            batch_text = {"input_ids": torch.as_tensor(texts_encoded["input_ids"]).unsqueeze(0).to(device), "attention_mask": torch.as_tensor(texts_encoded["attention_mask"]).unsqueeze(0).to(device)}
            
            class_embeddings = model.encode_text(batch_text) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == "__main__":

  

    test_0shot()