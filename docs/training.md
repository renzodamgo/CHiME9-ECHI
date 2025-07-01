# Training a Model

The script to train a model from scratch can be run using

```bash
python run_train.py
```

This is equivalent to running the following steps

```bash
python -m scripts.train.unpack
python -m scripts.train.train
```

This will unpack the data downloaded and then use them to trian a model.

## Unpack

With your downloaded dataset, stored in `data/chime9_echi`, you can specify
`model_sample_rate`, which is the sample rate input for your model, and a
`device` (`ha` or `aria`), this script will

- Resample the `device` audio and the corresponding refrences to
`model_sample_rate` and then segment it all in to just the speech segments,
- Resample the rainbow passages to `model_sample_rate`.

The outputs of this stage will be saved into `paths.working_dir` under
`train_segments` for the device and reference audio, and `participant` for the
rainbow passages by default. Parameters for this stage are found in
`config.train.unpack.yaml`.

If you want to use the same data segmentation for each training run, this
stage will only need to be run once. The `unpack` script will look for the
output files before processing, so if it is run a second time, it won't
produce any new files. If you wish to modify the unpack script to generate
different segments, it is recommended that you specify a new `working_dir` in
`config/paths.yaml`, otherwise it may find the previous files and not perform
the new process for all files.

This stage can be run on CPU or GPU.

## Training

The training loop will create a dataloader which loads the data above, and
then defines a model to train with the given training parameters. There are
multiple configs associated with this stage, stored in `configs.train`:

- `dataloading`: Stores the paths of the audio files to load, and dataloading
parameters.
- `train`: The parameters/details for the training cycle, including epochs,
learning rate, loss functions, etc.
- `model`: Stores parameters for how to prepare the audio for input to the
model and the parameters for the model architecture.
- `wandb`: Information for logging runs to
[Weights and Biases](https://wandb.ai/site/models/). Setting values to `null`
will stop any logging here.

This stage should only be run on GPU.

There are two main configuration files: `config/train/main_ha.yaml` and
`config/train/main_aria`, which correspond to training using the hearing aid
audio and Aria glasses audio, respectively.

## Example Usage

The default setting will train the baseline:

```bash
python run_train.py
```

If you have a custom model `your_enhancement_system`, you should store the
`torch.nn.Module` file in `src/shared/your_enhancement_system.py` and the
config in `config/train/your_enhancement_system.yaml`, using the default
`model.yaml` as a guide.

In `src/shared/core_utils.py`, add your model as an option in the `get_model`
function. Finally, to train your model, use the command

```bash
# If the data hasn't been unpacked
python run_train.py model=your_enhancement_name
# Elif the data has been unpacked
python scripts/train/train.py model=your_enhancement_name
```
