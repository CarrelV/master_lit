import torch
import inspect
from packaging import version
from torch.utils.data import DataLoader

from losses import CLIPMoCOLoss

def calculate_the_importance_label(model, data_loader, num_samples, device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
    
    loss_fn = CLIPMoCOLoss()
    

    for idx, batch in enumerate(data_loader):
        if idx >= num_samples:
            break

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
       
            
        output = model(image,text)
        
        loss = loss_fn(output,keys_for_this_batch)
       

        # inputs.pop("idx", None)
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(cuda_device)

        # return_dicts = model(**inputs)

        # loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

    return gradients_dict


def calculate_the_importance_expect(model, task, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, (ques_id, feats, boxes, sent, target) in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        # inputs.pop("idx", None)
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(cuda_device)

        # return_dicts = model(**inputs)

        # logits = return_dicts["logits"]

        feats, boxes, target = feats.to(cuda_device), boxes.to(cuda_device), target.to(cuda_device)
        logits = model(feats, boxes, sent)

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


def compute_fisher(model, train_dataloader, num_samples):
    importance_method = calculate_the_importance_label

    # import pdb
    # pdb.set_trace()

    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)


    
    grad_type = "square"

    return importance_method(model, train_dataloader, num_samples, cuda_device, grad_type)