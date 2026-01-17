from argparse import ArgumentParser
import yaml
import torch

import clip

from terra_utils import load_terra

if __name__ == '__main__':
    parser = ArgumentParser(description="Object Retrieval Test")
    parser.add_argument(
        '--terra',
        type=str,
        help="/path/to/Terra.pkl object"
    )
    parser.add_argument(
        '--region_monitoring_tasks',
        type=str,
        help="/path/to/region_monitoring_tasks.yaml file of region monitoring tasks"
    )
    parser.add_argument(
        '--prediction_method',
        type=str,
        default="max",
        help="Region monitoring method: [max, thresh, mix, aib]"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help="K for running top-k with `max` method"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.23,
        help="Threshold for task relevance"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()
    
    terra = load_terra(args.terra)
    terra.alpha = args.alpha
    
    with open(args.region_monitoring_tasks, 'rb') as f:
        region_tasks = yaml.safe_load(f)["tasks"]
        
    # Encode prompts with CLIP
    tasks = [task["task"] for task in region_tasks]
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    print("\nCollected region tasks:", tasks)

    # Prediction regions given prompts
    terra.predict_regions(input_task_clip_tensor, tasks, args.prediction_method, K=args.k)
    
    # Display Results
    terra.display_terra()
    for task_idx in range(len(region_tasks)):
        terra.display_task_relevant_places(task_idx)
    terra.display_task_relevant_places()