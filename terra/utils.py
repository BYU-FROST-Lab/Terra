import torch.nn.functional as F
import re

def tensor_cosine_similarity(emb1, emb2):
    unscaled_logit_cos_sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
    return unscaled_logit_cos_sim  

def numeric_key(path):
        match = re.search(r"(\d+\.\d+)", path.stem)  # grabs the float in the name
        return float(match.group()) if match else float('inf')