# Baseline systems for the ECHI task of the CHiME-9 challange

## Sections

1. <a href="#install">Installing the software</a>
2. <a href="#data">Installing the dataset</a>
3. <a href="#preparation">Preparing the data</a>
4. <a href="#baseline"> Running the baseline</a>
5. <a href="#evaluation">Evaluating the output</a>
6. <a href="#results">Displaying performance metrics</a>
7. <a href="#troubleshooting">Troubleshooting</a>

## <a id="#install">1. Installing the software</a>

Clone this repository from GitHub

```bash
git clone git@github.com:CHiME9-ECHI/CHiME9-ECHI.git
cd CHiME9-ECHI
```

The installation of the necessary tools is detailed in `install.sh`.
We recommend to follow it step-by-step and adjust for your system if needed.
The script will build a conda environment called `echi_recipe`

```bash
install.sh
```

When running the system, remember to activate the conda environment and set the
necessary environment variables,

```bash
conda activate echi_recipe
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## <a id="data"> 2. Installing the data </a>

- download and install dataset
- set paths in config/paths/default

## 3. <a id="preparation">Preparing the data</a>

Prepare the `chime9_echi` dataset for use.

```bash
python scripts/prepare.py
```

## 4. <a id="baseline">Running the baseline</a>

The following script will run the baseline and produce output in the format
that is ready for evaluation.

```bash
bash scripts/make_dummy_submission.sh <chime9_echi_dir> <submission_directory>"
```

e.g.

```bash
bash scripts/make_dummy_submission.sh data/chime9_echi data/submission
```

## 5. <a id="evaluation">Evaluating the output</a>

Run the evaluate script

```bash
python scripts/evaluate.py submission=<submission_dir>
```

With test data

```bash
python scripts/evaluate.py submission=data/submission
```

The `decimate_factor` can be used to select every Nth segment for evaluation.
For example, to evaluate a subset of just 1/20 of the data

```bash
python scripts/evaluate.py submission=data/submission decimate_factor=20
```

## 6. <a id="results">Displaying performance metrics</a>

TODO

## 7. <a id="troubleshooting">Troubleshooting/a>

TODO
