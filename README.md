# Karaokey

Karaokey is a vocal remover that automatically separates the vocals and instruments. A deep learning model based on LSTMs has been trained to tackle the source separation. The model learns the particularities of music signals through its temporal structure.

## Dataset used to train the model
The models are trained using the MUSDB18 dataset, here is the link to [MUSDB18 dataset](https://doi.org/10.5281/zenodo.1117372).
More information about the dataset can also be found at [sigsep.io](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all dependencies needed for Karaokey.

### For latest dependencies run

```bash
pip install -r requirements.txt
```

### For stable dependencies run

```bash
pip install -r requirements-versions.txt
```

### For latest dependencies for gpu run

```bash
pip install -r requirements-gpu.txt
```



## Deployment

The app is deployed at https://karaokey.co.uk.
