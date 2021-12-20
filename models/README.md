# Morpheus Models

Pretrained models for Morpheus as well as accompanying documentation.

## Git LFS

The large model files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). Before cloning this repos, you must install the Git LFS command extension:
```
# apt update
# apt install git-lfs
```

## Repo Structure
Every Morpheus use case has a subfolder, named by the use case. These subfolders contain the model files for the use cases.  

The `triton_model_repo` contains the necessary directory structure and configuration files in order to run the Morpheus Models in Triton Inference Server. This includes symlinks to the above-mentioned model files along with corresponding Triton config files (`.pbtxt`). More information on how to deploy this repository to Triton can be found in the [README](./triton-model-repo/README.md).

Models can also be published to an [MLflow](https://mlflow.org/) server and deployed to Triton using the included deployment plugin in `mlflow-triton-plugin`. More information on configuration and usage can be found [here](./mlflow-triton-plugin/README.md).

There is also a directory named `training-tuning-scripts` that contains training and fine-tuning scripts (`.py`) for the available pre-built use cases as well as a Jupyter notebook (`.ipynb`) version of the script.

In the root directory, the file `model-information.csv` contains the following information for each model:

 - **Model name** - Name of the model
 - **Use case** - Specific Morpheus use case the model targets
 - **Owner** - Name of the individual who owns the model
 - **Version** - Version of the model (major.minor.patch)
 - **Training epochs** - Number of epochs used during training
 - **Batch size** - Batch size used during training
 - **GPU model** - Family of GPU used during training
 - **Model accuracy** - Accuracy of the model when tested
 - **Model F1** - F1 score of the model when tested
 - **Small test set accuracy** - Accuracy of model on validation data in datasets directory
 - **Memory footprint** - Memory required by the model
 - **Input data** - Typical data that is used as input to the model
 - **Thresholds** - Values of thresholds used for validation
 - **NLP hash file** - Hash file for tokenizer vocabulary
 - **NLP max length** - Max_length value for tokenizer
 - **NLP stride** - stride value for tokenizer
 - **NLP do lower** - do_lower value for tokenizer
 - **NLP do truncate** - do_truncate value for tokenizer
 - **Version CUDA** - CUDA version used during training
 - **Version Python** - Python version used during training
 - **Version Ubuntu** - Ubuntu version used during training
 - **Version Transformers** - Transformers version used during training

## Current Use Cases Supported by Models Here
### Sensitive Information Detection
Sensitive information detection is used to identify pieces of sensitive data (e.g., AWS credentials, GitHub credentials, passwords) in unencrypted data. The model for this use case is an NLP model, specifically a transformer-based model with attention (e.g., mini-BERT).

### Anomalous Behavior Profiling
This use case is currently implemeted to differentiate between crypto mining / GPU malware and other GPU-based workflows (e.g., ML/DL training). The model is a XGBoost model.

### Phishing Email Detection
This use case is currently implemeted to differentiate between phishing and non-phishing emails. The models for this use case are NLP models, specifically transformer-based models with attention (e.g., BERT).

### Humans-As-Machines-Machines-As-Humans Detection
This use case is currently implemeted to detect changes in users' behavior that incate a change from a human to a machines or a machine to a human. The model is an ensemble of an autoencoder and fast fourier transform reconstruction.


