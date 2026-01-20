import os
import re
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

# LEVEL = {
#     "OBJECT": 0,
#     "PLACE": 1,
#     "REGION": 2 # regions = 2+
# }

def tensor_cosine_similarity(emb1, emb2):
    unscaled_logit_cos_sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
    return unscaled_logit_cos_sim  

def chunked_tensor_cosine_similarity(
    emb1,               # (N, D)
    emb2,               # (M, D)
    chunk_size=8192,
):
    """
    Yields (start_idx, cosine_similarity_chunk)
    cosine_similarity_chunk shape: (chunk, M)
    """
    N = emb1.shape[0]
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            emb1_chunk = emb1[start:end]  # (chunk, D)

            # Same operation as original, just chunked
            cos_sim = F.cosine_similarity(
                emb1_chunk.unsqueeze(1),  # (chunk, 1, D)
                emb2.unsqueeze(0),        # (1, M, D)
                dim=2
            )  # (chunk, M)

            yield start, cos_sim


def numeric_key(path):
        match = re.search(r"(\d+\.\d+)", path.stem)  # grabs the float in the name
        return float(match.group()) if match else float('inf')

def int_defaultdict():
    return defaultdict(int)

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