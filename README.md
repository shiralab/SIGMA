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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


