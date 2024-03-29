# mlops_project

This is the project repo for the DTU course Machine Learning Operations (02476) for group 12.

## Project Description

The goal of this project is to develop an ML system focusing on classifying dogs using their images. The project will utilize Docker to ensure a consistent and isolated environment for development and deployment. We will use Nicki Skaftes Cookiecutter template for project structuring and organization for a good workflow. Specifying hyperparameters is also crucial for reproducibility so therefore we will also use configuration management with Hydra. Weights and Biases will also be integrated for comprehensive experiment tracking and logging and then Pytorch Lightning will be used to reduce boilerplate code. The dataset consists of 20.000 images of dogs with labels, which can be used for image classification. Since the data takes up a lot of space, we will use DVC to efficiently manage and version the data, as well as handling our storage requirements. The models used for this project will be a custom CNN model for a baseline and then a ViT as a more advanced model. The Transformers framework by Hugging Face will be used to integrate the ViT model.


## Technologies/Frameworks

A list of frameworks the projects aims to include:

|   Framework  | Function |
| -------- | ------- |
| Git   | Code Versioning    |
| DVC    | Data Versioning   |
| Conda    | Virtual Environment    |
| Cookiecutter   | Project template   |
| Docker   | Create shareable environment  |
| Hydra   | Config for Hyperparameters & Model  |
| Tensorboard | Profiling |
| W&B   | Track and log values for experiments  |
| Pytorch-lightning   | Reduce boilerplate Code |
| Pytorch   | Deep learning framework |
| timm   | Pre-trained models |
| FastAPI |    |


All codes should be run from the root directory of the project, so make sure to path to the desired file from the `mlops_project` folder. 

## Data

In this project we are using the `Stanford Dogs Dataset` which contains 120 breeds in 20,580 images of dogs with labels and bounding boxes. The dataset was chosen due to it's simplicity and relatively large number of images. We have decided to use a subset of the dataset (10 breeds) to minimize trainingtimes, however, we have made sure that it is simple to train on the entire dateset using the `data_config.yaml` file where you can specify which breeds to run using the `classes` parameter.

To download data locally run:

`dvc remote add -f -d storage gs://mlops-data-dog/` 

then:

`dvc pull` 

### Data loader

`make_data.py` is a script for preparing image datasets, supporting both complete and sparse dataset creation.
- **Complete Dataset**: Processes all images and annotations.
- **Sparse Dataset**: Choose classes to process
The selection is done through `src/config/data/data_config.yaml` at `classes`. 

## Trainer

`train_model.py` is a script from which you can train your models using the specific hyperparemeter and architecture chosen by the user. In the end you should see a file named according to the following template: `<model>_<Timestamp>` which should help keep track of multiple trained models. 

To change the trained model go to `model_config.yaml` and specify your desired architecture. Afterwards change the `default_model` parameter in `model_config.yaml` to the name of the model you wish to train.

## Predicter

`predict_model.py` is a script from which you can pass any image, and predict class of dog in the picture. You can choose a specific model by inserting the file name in `model_config.yaml` under the parameter `WhatEverThisParameterIsNamed!!`. 


## Docker

To build the docker image to train the model in `train_model.py` use the following command:

`docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest`

In order to run the trainer on a docker image it is recommended to use the following command:

`docker run --shm-size=1g --name <experiment_name> -e WANDB_API_KEY=<insert_API_KEY> trainer:latest`

Remember to replace `<experiment_name>` with the desired name for the run and `<insert_API_KEY>` with your WANDB API key.



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── app                  <- Application for running the inference model
├── config
│   ├── data             <- Contains config for data loading
│   ├── model            <- Contains config for model
│   └── config.yaml      <- Combines config for data and model
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   ├── index.md         <- Homepage for your documentation
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
