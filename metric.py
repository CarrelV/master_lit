import torch
from imagenetv2_pytorch import ImageNetV2Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config as CFG
from robustness.tools.imagenet_helpers import common_superclass_wnid,ImageNetHierarchy
from utils import read_imagenet_class
from robustness import datasets
from dataloader import get_dataloader



######################## IMAGENET 0 SHOT ###############


def imagenet_0shot(model,tokenizer,dataset,device,printing=True):

    imagenet_prompt =[
        'a photo of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        '{} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]


    transform_ImageNet = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.444, 0.421, 0.385), 
                                     (0.285, 0.277, 0.286)),
                ])
    
    if printing:
        print("-------------------------\n")

    if dataset == "all":
        

        ds = ImageNetV2Dataset(transform=transform_ImageNet)
        test_loader = DataLoader(ds, batch_size=CFG.test_batch_size, num_workers=CFG.num_workers)

        imagenet_classes = read_imagenet_class()
            
        if printing:
            print(f"On the complete test set with {len(imagenet_classes)} different classes")

    elif dataset == "tiny":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid = common_superclass_wnid('mixed_13')

        
        class_ranges, labels_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        if printing:
            print(f"On a massively reduced test set with {len(labels_map)} different classes, less complex, representative of common object")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "big":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(400,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        if printing:
            print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "medium":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(100,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        if printing:
            print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    elif dataset == "small":

        in_path = "ImageNet"
        in_info_path = "imageNet_utils"
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid, class_ranges, labels_map = in_hier.get_superclasses(25,
                                             balanced=False)

        custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=CFG.num_workers,
                                                        batch_size=CFG.batch_size,only_val=True)

        if printing:
            print(f"On a reduced test set with {len(labels_map)} different classes, where we grouped the classes by subclasses")
        imagenet_classes = [item.split(",")[0] for item in labels_map.values() ]

    if printing:
        print("\n-------------------------")

        print("Some example of classes:")
        print(imagenet_classes[:10])

        print("\n")

    print("compute text weight")
    test_fct()
    #text_zeroshot_weight = compute_text_weight_zeroshot(model=model,tokenizer=tokenizer,device=device,classnames=imagenet_classes,template=imagenet_prompt)
    text_zeroshot_weight = compute_text_weight_zeroshot(classnames=imagenet_classes,template=imagenet_prompt)

    print("done compute text weight")
    counter = 0
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in test_loader:
            
          
            
            #images = feature_extractor(images,return_tensors="pt")

            #images = images["pixel_values"]
            
            images = images.to(device)
            target = target.to(device)
            
            # predict
            image_features = model.module.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_zeroshot_weight

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 
    
    if printing:
        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}\n")

    return top1,top5


def test_fct():
    print("I got into text")

#def compute_text_weight_zeroshot(model,tokenizer,device,classnames, templates):

def compute_text_weight_zeroshot(classnames,templates):
    print("1")
    with torch.no_grad():
        zeroshot_weights = []
        print("2")
        for classname in classnames:
            print("3")
            texts = [template.format(classname) for template in templates]  #format with class

            print("4")
            texts_encoded = tokenizer(texts,padding="max_length",max_length=CFG.max_length) #tokenize
            print("5")
            batch_text = {"input_ids": torch.as_tensor(texts_encoded["input_ids"]).to(device), "attention_mask": torch.as_tensor(texts_encoded["attention_mask"]).to(device)}
            print("about to encode text")
            class_embeddings = model.module.encode_text(batch_text) #embed with text encoder
            print("did encode text")

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights



#########################  FLICKR I2T and T2I RETRIEVAL ########################


def i2t_t2i_retrieval(model,dataset,tokenizer,feature_extractor,device,printing=True):


    test_loader = get_dataloader(dataset=dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,rank=0,world_size=1,batch_size=CFG.test_batch_size,shuffle=False,num_workers=CFG.num_workers,split="test")

    with torch.no_grad():

        top1_i2t, top5_i2t,top10_i2t,top1_t2i,top5_t2i,top10_t2i, n = 0., 0., 0. ,0. ,0. ,0. ,0.
        for batch in test_loader:

            image = batch["image"].to(device)
            text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}

            #print(text["input_ids"].shape)
            #print(text["attention_mask"].shape)
            # compute image_features
            image_features = model.module.encode_image(image)
            # For CLIP only
            #image_features = model.get_image_features(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # compute text features
            text_features = model.module.encode_text(text)
            # For CLIP only
            #text_features = model.get_text_features(**text)
            
            
            text_features /= text_features.norm(dim=-1, keepdim=True)

            

            sim_i2t = 100. * image_features @ text_features.T

            target = torch.eye(sim_i2t.shape[0]).to(device)
            
            target = target.argmax(dim=0)


            # measure accuracy
            acc1, acc5, acc10 = accuracy(sim_i2t, target, topk=(1, 5, 10))
            top1_i2t += acc1
            top5_i2t += acc5
            top10_i2t += acc10

            acc1, acc5, acc10 = accuracy(sim_i2t.T, target, topk=(1, 5, 10))
            top1_t2i += acc1
            top5_t2i += acc5
            top10_t2i += acc10

            n += image.size(0)

    top1_i2t = (top1_i2t / n) * 100
    top5_i2t = (top5_i2t / n) * 100 
    top10_i2t = (top10_i2t / n) * 100
    

    if printing:
        print(f"Image 2 Text Retrieval on {dataset}:")
        print(f"Top-1 accuracy: {top1_i2t:.2f}")
        print(f"Top-5 accuracy: {top5_i2t:.2f}")
        print(f"Top-10 accuracy: {top10_i2t:.2f}")

    top1_t2i = (top1_t2i / n) * 100
    top5_t2i = (top5_t2i / n) * 100 
    top10_t2i = (top10_t2i / n) * 100 

    if printing:
        print(f"Text 2 Image Retrieval on {dataset}:")
        print(f"Top-1 accuracy: {top1_t2i:.2f}")
        print(f"Top-5 accuracy: {top5_t2i:.2f}")
        print(f"Top-10 accuracy: {top10_t2i:.2f}")

    return top1_i2t,top5_i2t,top1_t2i,top5_t2i


############################ ACCURACY #######################

def accuracy(output, target, topk=(1,)):
    indices = output.topk(max(topk), 1, True, True)[1].t()
    correct = indices.eq(target.expand_as(indices))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

