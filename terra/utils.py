import os
import re
import numpy as np

import torch.nn.functional as F

def tensor_cosine_similarity(emb1, emb2):
    unscaled_logit_cos_sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
    return unscaled_logit_cos_sim  

def numeric_key(path):
        match = re.search(r"(\d+\.\d+)", path.stem)  # grabs the float in the name
        return float(match.group()) if match else float('inf')

def find_latest_itr(folder: str):
    """Finds the largest iteration number."""
    regex = re.compile(r"ptxpt_pc_dict_itr(\d+)\.pkl")
    max_itr = -1
    for f in os.listdir(folder):
        match = regex.search(f)
        if match:
            itr = int(match.group(1))
            if itr > max_itr:
                max_itr = itr
    return max_itr

def find_latest_file(folder: str, pattern=r"ptxpt_pc_dict_itr(\d+)\.pkl"):
    """Finds the ptxpt_pc_dict_itr{last_itr}.pkl file with the largest iteration number."""
    regex = re.compile(pattern)
    max_itr, latest_file = -1, None
    for f in os.listdir(folder):
        match = regex.search(f)
        if match:
            itr = int(match.group(1))
            if itr > max_itr:
                max_itr = itr
                latest_file = f
    return latest_file

def random_color():
    return np.random.rand(3).tolist()  # Generates a random RGB color