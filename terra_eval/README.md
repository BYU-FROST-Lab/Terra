# Terra Evaluation
This repository contains code to aid in evaluating Terra 3DSGs


# Table of Contents
- [Terra Evaluation](#terra_eval)
- [Table of Contents](#table-of-contents)
- [Task-Execution with Terra](#task-execution-with-terra)


# Task-Execution with Terra


<details open>

<summary><b>Simulated Object Retrieval Experiment</b></summary>

This experiment is done as follows:
- Modify the `terra_eval/config/sim/<test>_object_experiment.yaml` file to match the build Terra dataset location and output folders.
- Run the object retrieval experiment as follows:
```bash
python3 -m terra_eval.sim_object_experiment --params=terra_eval/config/sim/<test>_object_experiment.yaml
```
- This should print out results that match those in the ICRA 2026 Published Paper.
</details>

