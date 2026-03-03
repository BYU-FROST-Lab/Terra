from argparse import ArgumentParser
import yaml
import torch

import clip

from terra.utils import load_terra

if __name__ == '__main__':
    parser = ArgumentParser(description="Object Retrieval Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/region_monitoring.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()

    with open(args.params, 'rb') as f:
        object_task_params = yaml.safe_load(f)
    
    terra = load_terra(object_task_params["terra"])
    terra.alpha = object_task_params["alpha"]
    object_tasks = object_task_params["object_tasks"]
        
    # Encode prompts with CLIP
    tasks = [task["task"] for task in object_tasks]
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1, keepdim=True))
    print("\nCollected object tasks:", tasks)
    
    # Prediction objects given prompts
    terra.predict_objects(
        input_task_clip_tensor, 
        tasks[terra.num_terrain:], 
        object_task_params["prediction_method"],
        object_task_params["trim"]
    )
    
    # Display Results
    terra.display_terra()
    terra.display_terra(display_pc=True, color_pc_clip=False, color_terrain=False)
    terra.display_terra(display_pc=True, plot_objects_on_ground=True, color_pc_clip=False, color_terrain=False)