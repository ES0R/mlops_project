# mlops_project

This is the project repo for the DTU course Machine Learning Operations (02476) for group 12.

## Project Description

The goal of this project is to develop an ML system focusing on classifying dogs using their images. The project will utilize Docker to ensure a consistent and isolated environment for development and deployment. We will use Nicki Skaftes Cookiecutter template for project structuring and organization for a good workflow. Specifying hyperparameters is also crucial for reproducibility so therefore we will also use configuration management with Hydra. Weights and Biases will also be integrated for comprehensive experiment tracking and logging and then Pytorch Lightning will be used to reduce boilerplate code. The dataset consists of 20.000 images of dogs with labels, which can be used for image classification. Since the data takes up a lot of space, we will use DVC to efficiently manage and version the data, as well as handling our storage requirements. The models used for this project will be a custom CNN model for a baseline and then a ViT as a more advanced model. The Transformers framework by Hugging Face will be used to integrate the ViT model.


## Technologies

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

## Data

Run `dvc remote add -f -d storage gs://mlops-data-dog/` then `dvc pull`. 

### Data loader

`make_data.py` is a script for preparing image datasets, supporting both complete and sparse dataset creation.
- **Complete Dataset**: Processes all images and annotations.
- **Sparse Dataset**: Choose classes by name or index from `class_mapping.yaml`


## Usage
```bash
# Process the complete dataset
python ./src/data/make_dataset.py --dataset complete

# Process a sparse dataset with specific classes
python ./src/data/make_dataset.py --dataset sparse --classes 0 1 "Airedale"
```

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
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
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
