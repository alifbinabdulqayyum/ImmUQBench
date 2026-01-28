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
# Test Deterministic Models
Run `predict-deterministic.sh` file for testing the trained deterministic models.
```
bash predict-deterministic.sh
```
To test the models for In-Distribution scenarios, modify the `predict-deterministic.sh` accordingly. For example, to test the deterministic models on ImmunoVirus dataset in the In-Distribution scenario, modify the file like this:
```
datasource="Virus" # "Virus" "Bacteria" "Tumor"
targetsource="Virus" # "Virus" "Bacteria" "Tumor"
```
Similarly to test the deterministic models on ImmunoBacteria dataset, trained with ImmunoTumor dataset, modify the file like this:
```
datasource="Tumor" # "Virus" "Bacteria" "Tumor"
targetsource="Bacteria" # "Virus" "Bacteria" "Tumor"
```
# Test UQ Models
Run `predict.sh` file for testing the trained UQ models.
```
bash predict.sh
```
To test specific UQ models, modify the `predict.sh` file accordingly. For example, to test the SWAG models on ImmunoVirus dataset in the In-Distribution scenario, modify the file like this:
```
prob_model="swag" #"dvbll", "mcd", "la", "svdkl", "swag", "ts", "edl"
datasource="Virus" # "Virus" "Bacteria" "Tumor"
targetsource="Virus" # "Virus" "Bacteria" "Tumor"
```
Similarly to test the SWAG models on ImmunoBacteria dataset, trained with ImmunoTumor dataset, modify the file like this:
```
prob_model="swag" #"dvbll", "mcd", "la", "svdkl", "swag", "ts", "edl"
datasource="Tumor" # "Virus" "Bacteria" "Tumor"
targetsource="Bacteria" # "Virus" "Bacteria" "Tumor"
```
# Evaluate Models
Run `Evaluate.ipynb` file to evaluate all the models and generate comparative result figures.