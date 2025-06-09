# Baseline systems for the ECHI task of the CHiME-9 challange

**work in progress**

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

## Make a submission

The following script will make a dummy submission.

```bash
bash scripts/make_dummy_submission <chime9_echi_dir> <submission_directory>"
```

e.g.

```bash
bash scripts/make_dummy_submission data/chime9_data data/submission
```

## Evaluate a submission

- running the evaluate script

```bash
python scripts/evaluate.py enhanced=<path_to_enhanced> reference=<path to references>
```

With test data

```bash
python scripts/evaluate.py enhanced=test/test_samples/test1 reference=test/test_samples/test2
```
