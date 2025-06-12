# Baseline systems for the ECHI task of the CHiME-9 challange

## Sections

1. <a href="#install">Installing the software</a>
2. <a href="#data">Installing the dataset</a>
3. <a href="#baseline"> Running the baseline</a>
4. <a href="#configuration"> Configuring the baseline</a>
5. <a href="#troubleshooting">Troubleshooting</a>

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

## <a id="baseline">3. Running the baseline</a>

The baseline system can be run using

```bash
python run.py
```

This is equivalent to running the following steps

```bash
python -m scripts.prepare
python -m scripts.inference
python -m scripts.evaluate
python -m scripts.report
```

## <a id="configuration">4. Configuring the baseline</a>

TODO

Run the evaluate script

```bash
python scripts/evaluate.py evaluate.submission=<submission_dir>
```

With test data

```bash
python scripts/evaluate.py evaluate.submission=data/submission
```

The `decimate_factor` can be used to select every Nth segment for evaluation.
For example, to evaluate a subset of just 1/20 of the data

```bash
python scripts/evaluate.py evaluate.submission=data/submission
   evaluate.decimate_factor=20
```

## <a id="troubleshooting">5. Troubleshooting/a>

TODO
