# Training pipeline
This directory contains scripts and configurations for training machine learning models. The training pipeline is designed to be modular and flexible, allowing for easy customization and experimentation with different architectures, datasets, and hyperparameters. Running the training scripts typically involves specifying a configuration file that outlines the model architecture, training parameters, and dataset details. The training process includes data loading, preprocessing, model training, validation, and checkpointing.

## 1. Prerequisites
Before running the training scripts, ensure that you have the following prerequisites installed:
- Python 3.10
- Required Python packages (see `training/requirements.txt`)
- CUDA Toolkit 12.x, cuDNN 9.x, NVIDIA graphics card drivers
- Sufficient disk space for datasets and model checkpoints

## 2. Data Preparation
1. Acquire the datasets:
```bash
usage: get_data.py [-h] --output-dir OUTPUT_DIR

Download multiple datasets into a specified folder

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Path to the folder where datasets should be downloaded
```
Example usage:
```bash
python data_utils/get_data.py --output-dir ./datasets/
```
2. Run the dataset preparation:
```bash
usage: data_prep.py [-h] --train-csv TRAIN_CSV --val-csv VAL_CSV --train-dir TRAIN_DIR --val-dir VAL_DIR --output-dir OUTPUT_DIR [--n-folds N_FOLDS] [--img-size H W]

CheXpert 5-fold preprocessing

options:
  -h, --help            show this help message and exit
  --train-csv TRAIN_CSV
                        Path to the training CSV file
  --val-csv VAL_CSV     Path to the validation CSV file
  --train-dir TRAIN_DIR
                        Path to the training images directory
  --val-dir VAL_DIR     Path to the validation images directory
  --output-dir OUTPUT_DIR
                        Path to the output directory
  --n-folds N_FOLDS     Number of folds for GroupKFold (default: 5)
  --img-size H W        Image size after resizing (default: 224 224)
```
Example usage:
```bash
python data_utils/data_prep.py \
--train-csv ./datasets/chexpert/1/train.csv \
--val-csv ./datasets/chexpert/1/valid.csv \
--train-dir ./datasets/chexpert/1/train \
--val-dir ./datasets/chexpert/1/valid \
--output-dir ./datasets/train_5fold_study1/ \
--n-folds 5 \
--img-size 224 224
```

## 3. Training
For the entire training pipeline the following technologies are used:
- PyTorch Lightning for reproducibility, streamlining of code writing and debloating.
- Hydra for instantiating objects and experiment logging.
- Tensorboard/ClearML for result and experiment logging.
### 3.1. Training scripts
1. Datamodule config setup at `training/configs/data/datamodule.yaml`:
```yaml
_target_: training.src.data.datamodule.CXRDataModule
data_dir: ${paths.data_dir}chexpert/5fold_study1/fold1/ # Set this according to your dataset
batch_size: 128
num_workers: 12 # Set this according to your hardware
tile_size: [224, 224]
pin_memory: True
train_augs: True
val_augs: False
```
- *Note*: The dataset directory that you are setting in the datamodule config must match the training you're going to do on it.
2. Model config setup at `training/configs/model/efficientformer.yaml`:
```yaml
_target_: training.src.models.efficientformer.EfficientFormer

optimizer:
  _target_: torch.optim.Adam
  _partial_: True
  lr: 0.001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: True
  mode: min
  factor: 0.1
  patience: 10

freeze_backbone: False
```
3. Run the script for training:
```bash
python training/src/train.py
```
- *Note*: You can override parameters you set in the config files, for example change the dataset directory inside the run of the script. This works for every parameter in the configs, but to give an example: `python training/src/train.py data.data_dir=datasets/other_dataset/`
4. Hyperparameter optimization:
Set the config file up `training/hparams_search/grid_search.yaml`, Run the training script with hyperparameter optimization:
```bash
python training/src/train.py -m hparams_search=grid_search
```
- *Note*: The hyperparameter optimization is done using grid search (or other search supported by optuna)

## 4. Evaluation
*WORK IN PROGRESS*
