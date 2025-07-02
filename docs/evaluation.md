# Evaluation

This document outlines the process for evaluating enhanced signals for the ECHI challenge.

### Sections

1. <a href="#prepare">Preparing signals for evaluation</a>
2. <a href="#evaluate">Running full evaluation</a>
3. <a href="#partial">Rapid partial evaluation </a>
4. <a href="#processors">Using multiple processors</a>

---

## <a id="prepare">1. Preparing signals for evaluation</a>

The evaluation code processes the complete ECHI development set. Before running, the enhanced signals must be generated and stored together in a single directory.

- **If you used the provided enhancement stage**: If you have configured and run the enhancement stage from this repository (see [enhancement](enhancement.md)), your signals are ready for evaluation.

- **If you used your own enhancement code**: Please ensure that your output signals match the format and naming convention described on the
[submission page](https://www.chimechallenge.org/current/task2/submission) of the ECHI challenge site.

The evaluation pipeline includes a validation stage, which will report any issues with the signals and stop the process early if errors are found.

## <a id="evaluate">2. Running full evaluation</a>

> Note: Running the full evaluation on a single CPU core, as described in this section, will take a very long time. See the following sections for how to configure a partial evaluation or use multiple processors.

The evaluation stage can be run in full on the enhanced signals using

```bash
python run_evaluation.py shared.exp_name=<EXP_NAME> paths.enhancement_output_dir=<ENHANCED_DIR>
```

Where:

- `<EXP_NAME>` is a name of your own choosing that will be used for the output directories.
- `<ENHANCED_DIR>` is the directory where your enhanced signals are stored.

> Note: For all the tools, configuration options can be provided on the commandline,
> as shown above, or my editing the `.yaml` files in the `config/` directory.

The `run_evaluation.py` pscript performs several steps that can also be run individually:

```bash
# 1. Prepare the reference signal
python scripts/evaluation/setup.py

# 2. Validate the submitted signals
python scripts/evaluation/validate.py paths.enhancement_output_dir=<ENHANCED_DIR>

# 3. Prepare the submitted signals for evaluation
python scripts/evaluation/prepare.py shared.exp_name=<EXP_NAME> paths.enhancement_output_dir=<ENHANCED_DIR>

# 4. Run the evaluation (this can take a very long time)
python scripts/evaluation/evaluate.py shared.exp_name=<EXP_NAME>

# 5. Collate the results and generate reports 
python scripts/evaluation/report.py shared.exp_name=<EXP_NAME>
```

The pipeline will generate the following directory structure:

```sh
   
working_dir/          # Root set in the main.yaml config
 └── experiments
    ├── <EXP_NAME>
    │   ├── enhancement
    │   │   | 
    │   │   ...
    │   └── evaluation
    │       ├── hydra
    │       ├── reports
    │       ├── results
    │       └── segments
    │           ├── aria
    │           │   ├── individual
    │           │   └── summed
    │           └── ha
    │               ├── individual
    │               └── summed
    ├── <OTHER EXPS>
    │    |
    |.   ...
    ...
```

Results will appear in
`working_dir/experiments/<EXP_NAME>/evaluation/reports`.

Results are reported at three levels:

- **Device level**: `report.dev.<segment_type>.<device>._._.json` - i.e. accumulated over all sessions for each device (`aria` or `ha`)
- **Session level**: `report.dev.<segment_type>.<device>.<session>._.json` - i.e. for a specific session and given device.
- **Participant level**: `report.dev.<segment_type>.<device>.<session>.<PID>.json` - i.e. for a specific participant.

There are two different segment types evaluated:

- `individual`: A single stream containing the speech of one participant.
- `summed`: The signal obtained by summing the three conversational participant streams.

(For details and motivation see the [Evaluation Plan](XXX) on the main challenge web.)

So for the `dev` set which has 10 sessions, for each segment type, there will be

- 2 device level report files
- 20 (2 devices x 10 session)  session-level files
- 60 (2 devices x 10 session x 3 participants) participant-level files.

For the `dev` set (10 sessions), this results in 164 report files per segment type. The reports are JSON files containing a dictionary for each metric, with statistics like `weighted mean`, `mean`, `standard deviation`, etc. The `weighted mean` (weighted by segment length) is the primary value used for reporting results. An accompanying `.csv` file is also generated for each `.json` report, containing the raw per-segment metrics.

## <a id="partial">3. Rapid partial evaluation</a>

Running the full evaluation is computationally expensive. If you have access to a Slurm HPC or a powerful workstation
with many CPU threads then see the section below on [parallel processing](#processors). Alternatively during development, you can run a faster, partial evaluations in two ways:

1. Computing a subset of the metrics.

2. Processing a subset of the speech segments.

This is achieved by editing the configuration files or by providing options on the command line.

Below we first provide a general [overview](#hydra) of the system configuration and then provide
some [specific examples](#config_examples) for rapid evaluation.

### <a id="hydra">3.1 Configuration overview

The main evaluation configuration files are in the `config/evaluation` directory. Key files include:

- `main.yaml`: Main configuration, imports other configs.

- `evaluate.yaml`: Configuration for the evaluation stage.

- `metrics.yaml`: Defines the metrics used in evaluation.

Shared parameters are located in the main config directory:

- `paths.yaml`: Defines paths for data and outputs.

- `shared.yaml`: Shared parameters like dataset paths.

You can override any configuration parameter from the command line. For example:

```bash
# Run the full pipeline but disable GPU usage for the enhancement step
python run_evaluation.py dataset=dev enhance.use_gpu=false
```

### <a id="config_examples"> 3.2 Configuration examples

This section provides guidance on customizing the evaluation process using different configurations and subsets of data.

#### Using a subset of metrics

Versa computes evaluation metrics based on a configuration file, by default:

```bash
config/evaluation/metrics.yaml
```

This default includes both CPU-friendly signal metrics and GPU-reliant DNN-based metrics. If you only want the faster CPU-based metrics, use the simplified configuration:

```bash
config/evaluation/metrics_quick.yaml
```

To apply this configuration, specify it via the `evaluate.score_config` parameter:

```bash
python scripts/evaluation/evaluate.py shared.exp_name=<EXP_NAME> \
  evaluate.score_config=config/evaluation/metrics_quick.yaml
```

You may edit `metrics_quick.yaml` or create a custom Versa config to tailor the metric selection further.

#### Using a subset of signals

The full evaluation set contains a large number of segments. For quicker, approximate results, you can evaluate a fraction of the data using batching.

Use the `evaluate.n_batches` parameter to divide the dataset into equal-sized batches, and `evaluate.batch` to specify which batch to evaluate. If not set, the first batch is used by default.

**Example: Evaluate only 1/50th of the data (defaulting to the first batch):**

```bash
python scripts/evaluation/evaluate.py shared.exp_name=<EXP_NAME> evaluate.n_batches=50
```

**Example: Evaluate the second batch out of 50:**

```bash
python scripts/evaluation/evaluate.py shared.exp_name=<EXP_NAME> \
  evaluate.n_batches=50 evaluate.batch=2
```

> Tip: Batching works seamlessly with Hydra’s `--multirun` functionality, enabling parallel evaluation across processors. See [Using Multiple Processors](#processors).

#### Other partial evaluations

By default, the evaluation includes:

- Both Aria and hearing aid devices
- Both individual and summed segment types

You can reduce computation by filtering by device or segment type.

**Example: Evaluate only the summed segments from the Aria glasses:**

```bash
python scripts/evaluation/evaluate.py shared.exp_name=<EXP_NAME> \
  evaluate.devices='[aria]' \
  evaluate.segment_types='[summed]'
```

These filters can be combined with other options.

**Example: Evaluate only CPU-based metrics for 1/50th of the Aria summed segments:**

```bash
python scripts/evaluation/evaluate.py \
  shared.exp_name=<EXP_NAME> \
  evaluate.score_config=config/evaluation/metrics_quick.yaml \
  evaluate.n_batches=50 \
  evaluate.devices='[aria]' \
  evaluate.segment_types='[summed]'
```

These configuration options offer flexibility for rapid iteration, targeted debugging, or distributing load across compute resources.

## <a id="processors">4. Using multiple processors</a>

The evaluation stage can be run in parallel across multiple CPU cores using Hydra's multi-run feature.

A bash script called `run_evaluation_parallel.sh` has been provided to make this easier.
The script provides configuration for two different 'launchers':

- `slurm`: Default option. See `config/hydra/launcher/echi_submitit_slurm` and edit to match your HPC environment.
- `local`: Running on multiple processors on a local machine.

Full usage options are shown below

```sh
Usage: ./run_evaluation_parallel.sh <EXP_NAME> <ENHANCED_DIR> [OPTIONS] [N_BATCHES]
Runs all evaluation stages using a scheduler to parallelize the evaluation stage.

Arguments:
  EXP_NAME              Name for the experiment (used for output directories)
  ENHANCED_DIR          Directory containing enhanced signals
  N_BATCHES (optional)  Number of parallel processes to run for evaluation.
                        Defaults to 40 for slurm, 10 for local.

Options:
  --launcher TYPE       Launcher type: 'slurm' or 'local'. Defaults to 'slurm'.
  --dry-run            Show commands that would be run without executing them.
  -h, --help            Display this help message and exit.

Examples:
  ./run_evaluation_parallel.sh exp1 /path/to/enhanced                    # SLURM with 40 batches
  ./run_evaluation_parallel.sh exp1 /path/to/enhanced --launcher local 8 # Local with 8 batches
  ./run_evaluation_parallel.sh exp1 /path/to/enhanced --dry-run          # Show commands without running
  ```
