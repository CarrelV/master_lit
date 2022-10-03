import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from transformers import ElectraConfig, ElectraTokenizerFast,ElectraForMaskedLM, ElectraForPreTraining





################## ELECTRA text Tower ###########
# mask_prob = 0.15
# lr = 5e-4
# bs = 128
# steps = 10**6
# max_length = 128
# generator_size_divisor = 4
# disc_config = ElectraConfig.from_pretrained(f'google/electra-1-discriminator')
# gen_config = ElectraConfig.from_pretrained(f'google/electra-1-generator')
# # note that public electra-small model is actually small++ and don't scale down generator size 
# gen_config.hidden_size = int(disc_config.hidden_size/generator_size_divisor)
# gen_config.num_attention_heads = disc_config.num_attention_heads//generator_size_divisor
# gen_config.intermediate_size = disc_config.intermediate_size//generator_size_divisor
# hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-1-generator")


class ELECTRAModel(nn.Module):
  
    def __init__(self, generator, discriminator, hf_tokenizer):
        super().__init__()
        self.generator, self.discriminator = generator,discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
        self.hf_tokenizer = hf_tokenizer

    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        sentA_lenths (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)

        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
        # reduce size to save space and speed
        mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)
    
        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = masked_inputs.clone() # (B,L)
            generated[is_mlm_applied] = pred_toks # (B,L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone() # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
                                    device=input_ids.device)
        return attention_mask, token_type_ids


