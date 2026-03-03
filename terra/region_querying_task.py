from argparse import ArgumentParser
import yaml
import torch

import clip

from terra.utils import load_terra

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
    terra.alpha = region_task_params["alpha"]
    region_tasks = region_task_params["region_tasks"]
    
    # Encode prompts with CLIP
    tasks = [task["task"] for task in region_tasks]
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
    print("\nCollected region tasks:", tasks)

    #Show place nodes relevant to each task based on human annotation
    for task_idx, task in enumerate(region_tasks):
        print(f"\nDisplaying human annotated relevant place nodes for task: {task['task']}")
        terra.visualizer.display_selected_nodes(
            terra.terra_3dsg,
            task["place_nodes"],
            pc=terra.pc
        )


    # Prediction regions given prompts
    terra.predict_regions(
        input_task_clip_tensor, 
        tasks, 
        region_task_params["prediction_method"], 
        K = region_task_params["k"]
    )
    
    # Display Results
    terra.display_terra()
    for task_idx in range(len(region_tasks)):
        terra.display_task_relevant_places(task_idx, heatmap_mode=True)
    terra.display_task_relevant_places()