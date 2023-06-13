# Movie Genre Classifier Deployment Guide

This guide provides step-by-step instructions to deploy the Movie Classifier on a Linux or macOS machine.

## Prerequisites

1. Install pyenv
    ```bash
    curl https://pyenv.run | bash
    ```

    Add the commands to ~/.bashrc:
    ```bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    ```
    Then, if you have ~/.profile, ~/.bash_profile or ~/.bash_login, add the commands there as well. If you have none of these, add them to ~/.profile.

    To add to ~/.profile:

    ```bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
    echo 'eval "$(pyenv init -)"' >> ~/.profile
    ```

    To add to ~/.bash_profile:
    ```bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
    echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
    ```

    For Zsh:
    ```bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    ```

    Restart your shell:
    ```bash
    exec "$SHELL"
    ```

2. Install Python build dependencies
   Mac OS X:
    If you haven't done so, install Xcode Command Line Tools (xcode-select --install) and Homebrew. Then:
    ```bash
    brew install openssl readline sqlite3 xz zlib tcl-tk
    ```

    Ubuntu/Debian/Mint:
    ```bash
    sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

3. Install Python using pyenv

    List python versions
    ```bash
    pyenv install --list
    ```

    Install latest stable version, for this project we used version 3.11.3
    ```bash
    pyenv install 3.11.3
    ```

## Deployment Steps

### Setup Environment

1. Create a Virtual Environment, we named the environement movies but feel free to use a different name:
   ```bash
   pyenv virtualenv 3.11.3 movies
   ```

2. Activate the Virtual Environment:
   ```bash
   pyenv activate movies
    ```

3. Install Required Packages:
   ```bash
   pip install -r requirements.txt
   ```

### Train the model

1. Run the script to train the model. The script will automatically fetch the dataset and train the best model we found.
    ```bash
    python train_model.py
    ```

### Deploy the Flask App

1. Run the Flask Application:
   ```bash
   python movies.py
   ```

# Sending Requests

1. Send request, for example:
    ```bash
    curl -d "overview=A movie about penguins in Antarctica building a spaceship to go to Mars." -X POST http://localhost:8000
    ```

2. Example response
   ```bash
   {"genre":"Comedy"}
   ```

# Analysis

For the data analysis, model search and baseline models, see the jupyter-notebook.

# Deployment

1. Install Docker

2. Run tensorflow-serving docker with the best model
    ```bash
    docker run -t --rm -p 8501:8501 --mount type=bind,source=$(pwd)/models/model/,target=/models/model/ -e MODEL_NAME=model emacski/tensorflow-serving:latest-linux_arm64
    ```

3. Send a post request
    ```bash
    curl -d '{"instances": [[31.993208, 2.3410976, 2.3042428, 0.0, 0.0, 1.7489684, 0.0, 1.2123615, 1.1253109, 1.2693826]]}' \
    -X POST http://localhost:8501/v1/models/model:predict
    ```

    for model that takes input text
    ```bash
    curl -d '{"instances": [["test"]]}' \
    -X POST http://localhost:8501/v1/models/model:predict
    ```