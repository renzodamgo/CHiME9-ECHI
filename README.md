# Baseline systems for the ECHI task of the CHiME-9 challange

**work in progress**

## Installation

- installing code
- installing data
- running prepare script

## Evaluation

- running the evaluate script

```bash
python scripts/evaluate.py enhanced=<path_to_enhanced> reference=<path to references>
```

With test data

```bash
python scripts/evaluate.py enhanced=test/test_samples/test1 reference=test/test_samples/test
```

(This will change shortly to operate at a higher level of abstraction)
