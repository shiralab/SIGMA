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

## Details
### 1. Parameters
#### 1.1 main.py
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

#### 1.2 start_main.sh
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

#### 1.3 start_llm_service.sh
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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


