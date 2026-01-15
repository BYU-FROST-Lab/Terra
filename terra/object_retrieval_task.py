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
        '--prediction_method',
        default="ms_avg",
        help="Object retrieval method: [ms_avg, ms_max, 3dsg]"
    )
    parser.add_argument(
        '--object_tasks',
        type=str,
        help="/path/to/object_retrieval_tasks.yaml file of object relevant tasks"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()
    
    terra = load_terra(args.terra)
    
    with open(args.object_tasks, 'rb') as f:
        obj_tasks = yaml.safe_load(f)["tasks"]
        
    # Encode prompts with CLIP
    tasks = [task["task"] for task in obj_tasks]
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    print("\nCollected tasks:", tasks)
    
    # Prediction objects given prompts
    terra.predict_objects(input_task_clip_tensor, tasks[terra.num_terrain:], args.prediction_method)
    
    # Display Results
    terra.display_terra()
    terra.display_terra(display_pc=True)