from argparse import ArgumentParser
import os
import yaml
import torch
import clip
import matplotlib.pyplot as plt
import open3d as o3d
import pickle as pkl
from terra.terra_utils import load_terra
from terra.utils import numeric_key, random_color, find_latest_itr, find_latest_file

def map_clipid_to_globalpts(global_pc, pc_clip_dict):
    count_threshold = 2
    clipid_2_globalpts = {}
    for global_idx in range(global_pc.shape[0]):
        if global_idx in pc_clip_dict.keys():
            max_id, max_count = max(pc_clip_dict[global_idx].items(), key=lambda x: x[1])
            # Make sure max_count is more than some threshold
            if max_count < count_threshold:
                if -1 in clipid_2_globalpts.keys():
                    clipid_2_globalpts[-1].append(global_idx)
                else:
                    clipid_2_globalpts[-1] = [global_idx]    
            elif max_id in clipid_2_globalpts.keys():
                clipid_2_globalpts[max_id].append(global_idx)
            else:
                clipid_2_globalpts[max_id] = [global_idx]
        else:
            if -1 in clipid_2_globalpts.keys():
                clipid_2_globalpts[-1].append(global_idx)
            else:
                clipid_2_globalpts[-1] = [global_idx]
    return clipid_2_globalpts

if __name__ == '__main__':
    parser = ArgumentParser(description="Region Monitoring Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/region_querying.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()

    with open(args.params, 'rb') as f:
        region_task_params = yaml.safe_load(f)
    
    terra = load_terra(region_task_params["terra"])
    # terra.visualizer.level_offset = 0.0  # Set level offset for better visualization


    # Dispplay just LiDAR point cloud
    terra.display_point_cloud(ms_map=False)

    # Display MS Map
    terra.visualizer.num_terrains = 7
    terra.display_point_cloud(ms_map=True)
    
    # Display just place nodes
    terra.display_places(
        display_pc=False,
        plot_ids=False
    )

    # Display Place nodes with pc
    terra.display_3dsg()
