# Baseline systems for the ECHI task of the CHiME-9 challange

work in progress

## 1. Installation

Install the code

```bash
git clone ...

export PYTHONPATH=src  # ?
```

## 2. Install the data

- download and install dataset
- set paths in config/paths/default

## 3. Prepare the data

Prepare the `chime9_echi` dataset for use.

```bash
python scripts/prepare.py
```

## 4. Make a submission

The following script will make a dummy submission.

```bash
bash scripts/make_dummy_submission.sh <chime9_echi_dir> <submission_directory>"
```

e.g.

```bash
bash scripts/make_dummy_submission.sh data/chime9_echi data/submission
```

## 5. Evaluate a submission

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
