# Baseline systems for the ECHI task of the CHiME-9 challenge

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

To make the `PYTHONPATH` setting persistent across terminal sessions, you can add
 the `export` command to your shell's configuration file (e.g., `~/.bashrc` for
 bash or `~/.zshrc` for zsh).

## <a id="data"> 2. Installing the data </a>

[Specific instructions on how to download and install the CHiME-9 ECHi dataset will
be provided here once available.]

The baseline system expects the dataset to be placed in the `data/chime9_echi`
 directory by default. The paths to various subsets of the data (e.g., training,
 development, evaluation) are defined in the `config/paths.yaml` file. If you
 choose to place the data in a different location, you will need to update
 `config/paths.yaml` accordingly. However, adhering to the default directory
 structure (`data/chime9_echi`) is recommended for ease of use.

## <a id="baseline">3. Running the baseline</a>

The baseline system can be run using

```bash
python run.py
```

This is equivalent to running the following steps

```bash
python -m scripts.prepare
python -m scripts.enhance
python -m scripts.evaluate
python -m scripts.report
```

Results will appear in the reports directory defined in `config/paths.yaml`. Results
are reported at three levels:

- The device level, `report.dev.<device>._._.json` - i.e. accumulated over all
 sessions.
- The session level, `report.dev.<device>.<session>._.json` - i.e. for a specific
 session and given device.
- The participant level, `report.dev.<device>.<session>.<PID>.json` - i.e. for a
 specific participant within a session for a given device.

For the `dev` set there will be 2, 24 (2 x 12) and 72 (2 x 12 x 3) of these
 files respectively.

The reports are stored a dictionary with an entry for each metric. Each metric in
 turn is presented with a dictionary storing the `mean`, `standard deviation`,
 `standard error`, `min value`, `max value` and the `number of segments`.

## <a id="configuration">4. Configuring the baseline</a>

The system uses [Hydra](https://hydra.cc/) for configuration management.
 This allows for a flexible and hierarchical way to manage settings.

The main configuration files are located in the `config` directory:

- `main.yaml`: Main configuration, imports other specific configurations.
- `shared.yaml`: Shared parameters used across different scripts (e.g., dataset paths,
general settings).
- `prepare.yaml`: Configuration for the data preparation stage (`scripts/prepare.py`).
- `enhance.yaml`: Configuration for the enhancement stage (`scripts/enhance.py`).
- `evaluate.yaml`: Configuration for the evaluation stage (`scripts/evaluate.py`).
- `report.yaml`: Configuration for the reporting stage (`scripts/report.py`).
- `metrics.yaml`: Configuration for the metrics used in evaluation.
- `paths.yaml`: Defines paths for data, models, and outputs.

You can override any configuration parameter from the command line.

For `run.py`, which executes the entire pipeline:

```bash
# Example: Run with a specific dataset configuration and disable GPU usage
# for enhancement
python run.py shared.dataset=my_custom_dataset enhance.use_gpu=false
```

For individual scripts like `scripts/evaluate.py`:

```bash
# Example: Evaluate a specific submission directory
python scripts/evaluate.py evaluate.submission=<submission_dir>

# Example: Evaluate with specific test data
python scripts/evaluate.py evaluate.submission=data/submission
```

Key configurable parameters include:

- **Dataset:** `shared.dataset` allows you to specify different dataset configurations.
- **Device Settings:** Parameters like `enhance.use_gpu` (true/false) and
 `enhance.device` (e.g., 'cuda:0', 'cpu') control hardware usage.
- **Evaluation:**
  - `evaluate.submission`: Path to the enhanced audio or transcriptions to be evaluated.
  - `evaluate.n_batches`, `evaluate.batch`: Control parallel processing during
 evaluation by splitting the data into batches.

## <a id="troubleshooting">5. Troubleshooting</a>

If you encounter issues, here are some common troubleshooting steps:

- **Activate Conda Environment:** Ensure your Conda environment (`echi_recipe`) is
 activated:

  ```bash
  conda activate echi_recipe
  ```

- **Check PYTHONPATH:** Verify that your `PYTHONPATH` environment variable is correctly
 set to include the `src` directory of this project:

  ```bash
  export PYTHONPATH=$PWD/src:$PYTHONPATH
  # (or ensure this is in your .bashrc or equivalent shell startup script)
  echo $PYTHONPATH
  ```

- **Verify Data Paths:** Double-check that the dataset paths in `config/paths.yaml`
 match the actual location of your CHiME-9 ECHi data. The default expected location
 is `data/chime9_echi`.
- **Hydra Log Files:** For detailed error messages and execution logs, inspect the
 Hydra log files. These are typically found in the `exp/<experiment_name>/hydra/`
 directory (e.g., `exp/main/hydra/`). The exact path will be printed at the start
 of a run.
- **Common Python Issues:** Check for common Python package installation problems
 or version conflicts within the Conda environment. Sometimes, reinstalling a
 problematic package can help.

### Running Evaluation in Batches

The following command is an example of how to run the evaluation stage in parallel
batches using Hydra's multirun feature and using either Hydra's submitit job launcher
plugin.

For running on a local machine with multiple cores,

```bash
python run.py evaluate.n_batches=10 evaluate.batch='range(1,11)' \
 hydra/launcher=echi_submitit_local  --multirun
```

For running on an HPC facility with a Slurm scheduler

```bash
python run.py evaluate.n_batches=200 evaluate.batch='range(1,201)' \
 hydra/launcher=echi_submitit_slurm  --multirun
```

- `evaluate.n_batches=10`: This parameter informs the script that the data should
 be conceptually divided into 10 batches.
- `evaluate.batch='range(1,10)'`: This specific Hydra syntax tells the system to
 launch multiple runs, iterating through the values generated by `range(1,10)`.
 In Python, `range(1,10)` produces numbers from 1 up to (but not including) 10,
 so this will create runs for batch numbers 1, 2, 3, 4, 5, 6, 7, 8, and 9. Each of
 these runs will process its corresponding segment of the data.
- `--multirun`: This is a Hydra flag that enables launching multiple jobs based on
 the sweep defined by `evaluate.batch`. These jobs may run sequentially or in
 parallel, depending on your Hydra launcher configuration (e.g., basic local
 launcher vs. a Slurm or other HPC scheduler launcher).

**Note on batch numbering:** If you intend to process all 10 batches, numbered for
 example from 1 to 10, you would use `evaluate.batch='range(1,11)'`.

If using an HPC facilty and Slurm, please check the configuration file
 `config/hydra/launcher/echi_submitit_slurm.yaml` and edit to fit your system.
