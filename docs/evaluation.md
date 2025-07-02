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
[submission page](XXX) of the ECHI challenge site.

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

Running the full evaluation is computationally expensive. During development, you can run a faster, partial evaluation in two ways:

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

TODO: complete this section

#### Using a subset of metrics

#### Evaluating a subset of signals

## <a id="processors">4. Using multiple processors</a>

The evaluation stage can be run in parallel across multiple CPU cores using Hydra's multi-run feature.

To run on a local machine with multiple cores, splitting the task into 10 batches:

```bash
python run_evaluation.py evaluate.n_batches=10 evaluate.batch='range(1,11)' \
 hydra/launcher=echi_submitit_local  --multirun
```

To run on an HPC cluster with a Slurm scheduler, splitting the task into
200 batches:

```bash
python run_evaluation.py evaluate.n_batches=200 evaluate.batch='range(1,201)' \
 hydra/launcher=echi_submitit_slurm  --multirun
```

- `evaluate.n_batches=10`: Informs the script to divide the data into 10 conceptual batches.
- `evaluate.batch='range(1,11)'`: This specific Hydra syntax tells the system to
 launch multiple runs, iterating through the values generated by `range(1,11)`.
 In Python, `range(1,11)` produces numbers from 1 up to (but not including) 11,
 so this will create runs for batch numbers 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10. Each of these runs will process its corresponding segment of the data.
- `--multirun`: This is a Hydra flag that enables launching multiple jobs based on
 the sweep defined by `evaluate.batch`. These jobs may run sequentially or in
 parallel, depending on your Hydra launcher configuration (e.g., basic local
 launcher vs. a Slurm or other HPC scheduler launcher).

**Note on batch numbering:** If you intend to process all 10 batches, numbered for
 example from 1 to 10, you would use `evaluate.batch='range(1,11)'`.

If using an HPC cluster with Slurm, please review and edit
 `config/hydra/launcher/echi_submitit_slurm.yaml` to match your system's configuration.gi
