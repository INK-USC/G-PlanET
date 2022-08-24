# G-PlanET

# RoboSense-methods


## Installation 

```bash 

conda create -n robosense python=3.8
conda activate robosense 

pip install -r requirements.txt

```

## File Structure

The `data` folder:

Included is the dataset we provide for G-PlanET, and the code for convert them to the format in iterative training and tapex training.

The `main` folder:

Includes all the code used in the main experiment

The `script` folder:

Includes all the script used in the main experiment

The `evaluation` folder:

The process needed to run our test framework and its code


We should install `transformers`, `torch-scatter`, `nltk`, `sacrebleu` for Tapex model.
Noted that you should use the command `pip install datasets "dill<0.3.5"` to install datasets and dill [issue link](https://github.com/huggingface/datasets/issues/4506)