from argparse import ArgumentParser
import yaml
import torch

import clip

from terra_utils import load_terra

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
    terra.visualizer.level_offset = 0.0  # Set level offset for better visualization
    
    # Display just place nodes
    terra.display_places(
        display_pc=False,
        plot_ids=False
    )

    # Display Places
    terra.display_places(
        display_pc=True,
        plot_ids=True
    )

    # Display just place nodes
    terra.display_places(
        display_pc=False,
        plot_ids=True
    )
