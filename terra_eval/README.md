# Terra Evaluation
This repository contains code to aid in evaluating Terra 3DSGs


# Table of Contents
- [Terra Evaluation](#terra_eval)
- [Table of Contents](#table-of-contents)
- [Task-Execution with Terra](#task-execution-with-terra)


# Task-Execution with Terra


<details open>

<summary><b>Object Retrieval Experiment</b></summary>

The following describes how to perform an object retrieval analysis on real-world data when no 3D ground truth is available.
This experiment is done as follows:
- Modify the `object_retrieval.yaml` file based on the parameters described below. Further modify the `object_tasks` based on your object tasks and the ground truth quantity of how many instances of each task there is in your dataset.
- Run the object retrieval experiment as follows. Note that `num_imgs` defines how many images for a single bounding box to display:
```bash
python3 -m terra_eval.object_retrieval_experiment --params=/path/to/object_retrieval.yaml --num_imgs=27
```
**Experiment Process**: After object prediction, this script will now display images with bounding boxes projected into the image frame first for accuracy and then for precision. Select `y` if across all images the predicted bounding box has a nonzero overlap with a unique ground truth object instance (not yet labeled as `y` for this metric (precision/recall)). The predicted object name is in the images' title. Select `n` if this criteria does not hold. Once completed for both accuracy and precision respectively, the computed metrics are printed out in the terminal. 

- YAML parameters are defined as:
    - `terra`: Path to the saved Terra 3DSG. 
        - *Note: `build_terra.py` saves a terra_3dsg.pkl and a Terra.pkl file. The first is just the nx.Graph 3DSG object and the latter is our Terra 3DSG class. This is asking for the latter.*
    - `object_tasks`: YAML list of object tasks
    - `prediction_method`: Pass a string of the method to use from the following [ms_avg, ms_max, 3dsg]. These methods are explained in detail in the paper. (Default: `ms_avg`)
    - `alpha`: Threshold to determine whether an object is task relevant (i.e. if its cosine-similarity score is above `alpha` then it is task-relevant). (Default: `0.23`) 
    - `trim`: Threshold used only for the MS-Trim method. (Default: `0.20`)

</details>


<details open>

<summary><b>Region Querying Experiment</b></summary>

T
</details>

