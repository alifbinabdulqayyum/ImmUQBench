# Installation Guide
Follow the installation guide from [VenusVaccine](https://github.com/ai4protein/VenusVaccine) repository.
# Data Processing & Download
Download the training data from [Google Drive](https://drive.google.com/drive/folders/1nM8e9_fJZJAo7ddJxLljijVnGDgLeJqH?usp=drive_link) (shared by the authors of [VenusVaccine](https://github.com/ai4protein/VenusVaccine))
# Train Deterministic Models
Run `train-deterministic.sh` for training deterministic models.
```
bash train-deterministic.sh
```
# Train UQ Models
Run `train.sh` for training UQ models.
```
bash train.sh
```
For training different UQ models change the following part of the `train.sh` file.
```
datasource="Virus" # "Virus" "Bacteria" "Tumor"
prob_model="sgld" # "edl", "la", "svdkl", "swag", "vbll", "sgld"
```
For example if you want train SWAG models on ImmunoBacteria dataset, change the `train.sh` accordingly:
```
datasource="Bacteria" # "Virus" "Bacteria" "Tumor"
prob_model="swag" # "edl", "la", "svdkl", "swag", "vbll", "sgld"
```