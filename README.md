# SIGMA: SHAP-Guided Implicit-Trajectory Generation for Metadata-Free LLM-Based AutoFE

SIGMA is a LLM-based AutoFE method that don't need the feature or task descriptions.
In addition, it also use an implicit trajectory to favour long-horizon optimization.

## Quick Start
### 1. Environment Build and Activation
1. Build conda environment from environment.yml
```
conda env create -f environment.yml -n SIGMA
```
2. Activate conda environment
```
conda activate SIGMA
```

### 2. Start LLMs service by vLLM
Use **start_llm_service.sh** to start model service.
The default model is Qwen/Qwen3-4B-Instruct-2507.

#### GPU Requirements
- **Minimum GPU memory required**: 20GB
- If you encounter Out-Of-Memory (OOM) errors, adjust the following parameters in `start_llm_service.sh`:
  - `--max-model-len`: Reduce the maximum model length to decrease memory usage
  - `--gpu-memory-utilization`: Reduce the GPU memory utilization ratio (default is 0.85)

#### Datatype Configuration
- The default datatype is `bfloat16` for optimal performance
- For older GPUs that do not support bfloat16, change `--dtype` to `float16` in the script
- **Note**: Using float16 instead of bfloat16 may influence the final results

#### Usage Examples
1. Default script will use GPU 0 and Port 8000
```
bash start_llm_service.sh
```
2. Script also support to specify GPUs.

**Single**:
```
bash start_llm_service.sh 2 # GPU 2 and Port 8002
```
**Mutiple**:
```
bash start_llm_service.sh 1 3 # Multiple Specific GPUs, GPU 1 3 and Port 8001 8003
```

### 3. Use bash scripts to process datasets
We use **start_main.sh** as the main entrance point.

- For a quick start, we can specify the dataset and seed. 
```
TARGET_DATASET=airlines TARGET_SEED=0 bash start_main.sh
```

- If don't set *TARGET_DATASET* and *TARGET_SEED*, the scripts will process all datasets with 3 seed splits.
```
bash start_main.sh
```

For the complete experiment, we recommend to use **nohup**, like the following.
```
nohup bash start_main.sh > run.log 2 >&1 &
```

Detailed results will be stored in the `results` directory.
You can check the final summary in the `results/main/xgboost/qwen3_gpu0/steps25_preset_coding/summary.csv`

## Notice on Reproducibility

Due to the inherent stochasticity of Large Language Models (LLMs), exact numerical reproduction of the reported results may not always be possible, even under identical experimental settings.

This variability may originate from several factors, including:
- stochastic token sampling during generation,
- non-deterministic GPU computation,
- backend inference differences across environments/frameworks,
- and sensitivity of iterative feature engineering trajectories.

To improve the robustness and reliability of evaluation, all experiments in this work were conducted using:
- 3 different random dataset splits (seeds),
- and 3 independent runs for each split.

The reported results are averaged across these repeated experiments.

Therefore, minor deviations in absolute performance metrics (e.g., ACC/F1/AUC) may be observed during reproduction. In such cases, we recommend evaluating reproducibility based on the consistency of overall trends, relative performance, and main conclusions rather than exact numerical matching.

## Details
### 1. Directory

#### 1.1 Overall Structure
```
SIGMA/
├── main.py                          # Main entrance point for feature engineering
├── start_main.sh                    # Bash script to run the entire pipeline
├── start_llm_service.sh             # Bash script to start vLLM service
├── combine_and_plot_trace.py        # Script to visualize and combine results
├── environment.yml                  # Conda environment configuration
├── README.md                        # This file
├── datasets/                        # Input datasets directory
├── results/                         # Output results directory
├── raw_results/                         # Raw output results for the paper
├── prompt_template/                 # LLM prompt templates
└── utils/                           # Utility modules
```

#### 1.2 datasets
- Contains all benchmark datasets for feature engineering evaluation
- Each subdirectory represents a different dataset (e.g., `airlines/`, `compass/`, `covertype/`, etc.)
- Each dataset contains 3 seed splits and a `metadata.json` file storing the basic information. 
- In each `seed_{n}` directory, it has 6 csv files, which are `raw_{set_name}` and `set_name`. For files start with `raw_`, it contains the original data, while others have masked feature name and encoded values.
- The datasets directory serves as input data for the SIGMA pipeline. For current appraoch, we use the masked input, which are the files of the name without `raw_`.
- Dataset format: tabular data ready for automated feature engineering

#### 1.3 results
- **Directory structure**: `results/main/xgboost/qwen3_gpu0/steps25_preset_coding/`
- Stores the output of feature engineering experiments
- Organized by model type (currently `xgboost/`), LLM model name (`qwen3_gpu0`), Experiments name `steps25_preset_coding`
- Contains performance metrics and generated features for each dataset.
- Introducing of resutlt files and directories
    - `code`: save the generated python code file.
    - `configs` : save the running configs
    - `logs`: running logs for each dataset
    - `prompt`: prompts and answers from LLMs
    - `stdout_logs`: output logs
    - `traces`: detailed generated information of each dataset
    - `prompt_backup`: Used prompt template
    - `script_backup`: Used main.py 
    - `summary.csv`: running results

### 2. Parameters

#### 2.1 main.py
This is the core Python script that runs the feature engineering pipeline.

**Required Arguments:**
- `--dataset_path` *(required)*: Path to a specific dataset and seed (e.g., `./datasets/airlines/seed_0`)

**Feature Engineering Parameters:**
- `--steps`: Number of feature generation steps (default: `10`)
- `--masked`: Use masked feature names in the dataset (default: `True`)
- `--reject_weak`: Whether to reject weak features (default: `False`)
- `--ablation_mode`: Ablation study mode (default: `ours`). Choices: `ours`, `random_grouping`, `no_grouping`, `no_noise`, `noise_sensitivity`, `rank_only`
- `--noise_level`: Noise level for feature grouping (default: `0.1`)

**LLM Configuration:**
- `--port`: Port number for LLM service (default: `8000`)
- `--prompt_path`: Path to the prompt template (default: `./prompt_template/user_template.txt`)
- `--model`: Model name to use (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--llm_preset`: LLM preset mode (default: `coding`). Choices: `debug`, `coding`, `math`
- `--temperature`: Temperature for LLM sampling (default: `None`)

**Model and Results:**
- `--task_model`: Task model type (default: `xgboost`)
- `--results_dir`: Directory to save results (default: `./results`)
- `--random_seed`: Random seed for reproducibility (default: `42`)
- `--track_test_curve`: Track test curve during optimization (default: `True`)

#### 2.2 start_main.sh
This bash script orchestrates the entire experiment pipeline, handling multiple datasets and seeds.

**Data Configuration:**
- `DATA_ROOT`: Root directory for datasets (default: `./datasets`)
- `SKIP_DATASETS`: Space-separated list of datasets to skip

**GPU/Service Configuration:**
- `LLM_GPUS`: GPU IDs for LLM service (default: `0`). Can specify multiple GPUs (e.g., `0 1 2`)
- `XGB_GPUS`: GPU IDs for XGBoost training (default: `0`). Can specify multiple GPUs
- `LLM_BASE_PORT`: Base port for LLM service (default: `8000`). Ports are calculated as `LLM_BASE_PORT + GPU_ID`

**Experiment Configuration:**
- `STEPS`: Number of feature generation steps (default: `25`)
- `MASKED`: Use masked feature names (default: `True`)
- `REJECT_WEAK`: Reject weak features (default: `False`)
- `TRACK_TEST_CURVE`: Track test curve (default: `True`)
- `LLM_PRESET`: LLM preset mode (default: `coding`)
- `TEMPERATURE`: Temperature for LLM sampling (optional)
- `RANDOM_SEED`: Random seed (default: `42`)
- `TASK_MODELS`: Space-separated list of task models (default: `xgboost`)
- `ABLATION_MODES`: Space-separated ablation modes (default: `ours`)
- `NOISE_LEVELS`: Space-separated noise levels (default: `0.1`)

**Quick Run Mode (For Reviewers):**
- `TARGET_DATASET`: Specific dataset to run (e.g., `jungle_chess_2pcs_raw_endgame_complete`)
- `TARGET_SEED`: Specific seed to run (e.g., `0`). When both are set, only this dataset+seed combination is processed

**Other Configuration:**
- `PY_SCRIPT`: Path to main.py script (default: `main.py`)
- `PROMPT_PATH`: Path to prompt template (default: `./prompt_template/user_template.txt`)
- `BASE_RESULTS`: Base results directory (default: `./results/main`)

**Example Usage:**
```bash
# Run single dataset with specific seed
TARGET_DATASET=airline TARGET_SEED=0 bash start_main.sh

# Run with custom GPUs and parameters
LLM_GPUS="0 1" XGB_GPUS="0" STEPS=50 bash start_main.sh

# Run in background with logging
nohup bash start_main.sh > run.log 2>&1 &
```

#### 2.3 start_llm_service.sh
This bash script starts the vLLM service for the LLM-based feature generation.

**Model Configuration:**
- `MODEL`: HuggingFace model ID (default: `Qwen/Qwen3-4B-Instruct-2507`)

**GPU Arguments (Positional):**
- Positional arguments specify which GPUs to use
- If no arguments are provided, defaults to GPU 0
- Example: `bash start_llm_service.sh 0 1 2` starts servers on GPUs 0, 1, and 2

**vLLM Server Parameters (in script):**
- `--seed`: Random seed for reproducibility (default: `42`)
- `--dtype`: Data type for model (default: `float16`). Can be `bfloat16` or `float16`
- `--gpu-memory-utilization`: GPU memory utilization ratio (default: `0.85`). Range: 0.0-1.0
- `--enable-prefix-caching`: Enable prefix caching for efficiency
- `--port`: Server port (automatically calculated as `8000 + GPU_ID`)
- `--max-model-len`: Maximum model sequence length (default: `65536`)

**Example Usage:**
```bash
# Single GPU (GPU 0)
bash start_llm_service.sh

# Specific GPU
bash start_llm_service.sh 2

# Multiple GPUs
bash start_llm_service.sh 0 1 2 3

# With environment variables to override model
MODEL="Qwen/Qwen3-4B-Instruct-2507" bash start_llm_service.sh 0
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


