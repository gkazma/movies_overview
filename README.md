# Movie Genre Classifier Deployment Guide

This guide provides step-by-step instructions to deploy the Movie Classifier on a Linux or macOS machine.

## Prerequisites
1. Create a /dataset folder at the root of the project directory

2. Download movies_metadata.csv dataset from [here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv) and put it inside the /dataset folder

3. Install conda on your machine by following the installation [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

4. Create a conda environment from the requirements file as follows:
    ```bash
    conda create --name <env> --file requirements.txt
    ```
5. Install docker by following these [instructions](https://docs.docker.com/engine/install/).
## Data Preparation
Prepare the data for training the model, i.e. data cleaning, data preparation and train/test splits by running the data_peparation.py script:
```bash
python data_preparation.py
```
See the jupyter notebook [data_preparation.ipynb](notebooks/data_preparation.ipynb) for the full data analysis and cleaning steps.
## Model Training
Train the best model we found by running the train_model.py script:
```bash
python train_model.py
```
See the jupyter notebook [model_search.ipynb](notebooks/model_search.ipynb) for the full model search and results.

## Deploy the model
We will use tensorflow-serving docker for serving the model (make sure you have Docker intalled on your machine, see Prerequisites). Run the TensorFlow Serving container pointing it to this model and opening the REST API port (8501):

On Linux or MacOS using intel chips:
```bash
docker run -t --rm -p 8501:8501 --mount type=bind,source=$(pwd)/models/model/,target=/models/model/ -e MODEL_NAME=model tensorflow/serving
```
On MacOS using M1 chips:
```bash
docker run -t --rm -p 8501:8501 --mount type=bind,source=$(pwd)/models/model/,target=/models/model/ -e MODEL_NAME=model emacski/tensorflow-serving:latest-linux_arm64
```
# Sending Requests

Send request, for example:
```bash
curl -d '{"instances": ["love love love love"]}' -X POST http://localhost:8501/v1/models/model:predict
```

Example response
```bash
{"predictions": [["Drama", "Romance"]]}
```
