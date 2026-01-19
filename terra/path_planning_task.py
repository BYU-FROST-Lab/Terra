from argparse import ArgumentParser
import yaml
import torch

import clip

from terra_utils import load_terra

if __name__ == '__main__':
    parser = ArgumentParser(description="Path Planning Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/path_planning.yaml file of path planning tasks"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()

    with open(args.params, 'rb') as f:
        path_planning_params = yaml.safe_load(f)
    
    terra = load_terra(path_planning_params["terra"])
    terra.alpha = path_planning_params["alpha"]
    
    queries = path_planning_params.get("queries", {})
    start_query = queries.get("start")  # None if not provided
    destination_query = queries["destination"]
    
    # Encode prompts with CLIP
    if start_query is not None:
        tasks = [start_query, destination_query]
    else:
        tasks = [destination_query]
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    print("\nCollected destination task", tasks)
    
    # Prediction objects given prompts
    terra.plan_path_to_destination(
        input_task_clip_tensor,
        terrain_preferences=path_planning_params["terrain_preferences"],
        method=path_planning_params["prediction_method"]
    )
    
    # Display Results
    # terra.display_terra()
    terra.display_path() # black = start, red = destination