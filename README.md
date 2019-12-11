# Karaokey

Karaokey is a vocal remover that automatically separates the vocals and instruments. A deep learning model based on LSTMs has been trained to tackle the source separation. The model learns the particularities of music signals through its temporal structure.

The model is trained using [PyTorch](https://pytorch.org/) and deployed using [Flask](https://palletsprojects.com/p/flask/).

## Dataset used to train the model
The models are trained using the MUSDB18 dataset, here is the link to [MUSDB18 dataset](https://doi.org/10.5281/zenodo.1117372).
More information about the dataset can also be found at [sigsep.io](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all dependencies needed for Karaokey.

### Installing libsndfile and ffmpeg
Libsndfile and ffmpeg is needed for audio processing.

**For linux run this command**
```bash
sudo apt install -y libsndfile1 ffmpeg 
```
**For windows**
To install libsndfile and ffmpeg we will have to use conda environment alternatively you can also follow the instructions on [libsndfile](http://www.mega-nerd.com/libsndfile/#Download).
```bash
conda install -c conda-forge ffmpeg libsndfile
```

### For latest dependencies run

```bash
pip install -r requirements.txt
```

### For stable dependencies run
The versions mentioned in this files are tried and tested. The application was tested on this versions.
```bash
pip install -r requirements-versions.txt
```

### For latest dependencies for gpu run
If you want to train the models on GPU please install these dependencies. Cuda based GPU dependencies are mentioned in this file, however if you want to install any other ones please have a look at [PyTorch's](https://pytorch.org/) installation guidelines.
```bash
pip install -r requirements-gpu.txt
```

> Note: If you get any error while running the scripts of application, please try and install the versions used "requirements-versions.txt" the versions mention in this file is tried and tested.

## Running the Flask Application
If you want to run Karaokey application using the trained model. Please follow following steps. Make sure you have all dependencies installed.
Setting the flask environment variables.
```bash
env FLASK_APP=./application/__init__.py 
```
Run flask app
```bash
flask run
```
