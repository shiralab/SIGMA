import argparse
import json
import logging
import os
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch.multiprocessing as tmp
from utils.data import get_X_y, load_datasets, mask_feature_names
from utils.cls_model import eval_with_model, eval_with_model_final
from utils.importance_score import (
    divide_groups_by_rank,
    divide_groups_by_shap,
    randomly_assign_groups,
)
from utils.prompt_process import (
    build_detailed_features_inform,
    build_feature_blocks,
    build_previous_report,
    build_top_k_report,
    fill_prompt_template,
)
from utils.llm import call_llm, extract_code_block
from utils.save import save_prompt_answer
from utils.run_code import save_code, apply_code
from utils.evaluate import eval_new_generated_features


DEFAULT_PROMPT_PATH = "./prompt_template/user_template.txt"
ABLATION_MODES = {
    "ours",
    "random_grouping",
    "no_grouping",
    "no_noise",
    "noise_sensitivity",
    "rank_only",
}
WEAK_MONITOR_MODES = {"ours", "noise_sensitivity", "no_noise", "rank_only"}
LLM_PRESETS = {"debug", "coding", "math"}
USE_WEAK_GROUP_IN_PROMPT = True
FROZEN_STEPS = {
    "top": 1,
    "useful": 2,
    "weak": 4
}

MIN_DELTA = 0.05
K_PATIENCE = 5
FORCE_END = 25

import re

def _eval_feature_worker(args):
    """
    在独立子进程中对单个特征做完整评估（含 CV）。
    子进程拥有自己的 CUDA context，GPU driver 做 time-multiplex。
    """
    tag, x_tr, x_va, x_te, y_train, y_val, task_model, parallel_cv, random_state = args
    clf, res = eval_with_model(
        x_tr, y_train, x_va, y_val,
        task_model=task_model,
        parallel_cv=parallel_cv,
        random_state=random_state,
    )
    return tag, clf, res

def parse_generated_features(generated_feat):
    if not isinstance(generated_feat, str) or not generated_feat.strip():
        return [], []

    text = generated_feat.lower()
    # Robust extraction: tolerate mixed styles like f56+f59, ratio(f56,f59), np.log(f56)_plus_f59.
    used_features = sorted(set(re.findall(r'\bf\d+\b', text)))

    raw_tokens = re.split(r'[^a-z0-9]+', text)
    used_operations = sorted({
        token for token in raw_tokens
        if token and not re.fullmatch(r'f\d+', token) and not token.isdigit()
    })

    return used_features, used_operations


def parse_features_from_code(code_text):
    if not isinstance(code_text, str) or not code_text.strip():
        return []

    text = code_text.lower()
    return sorted(set(re.findall(r'\bf\d+\b', text)))

def operations_prompt_info(op_history):
    sorted_op_history = sorted(op_history.items(), key=lambda x: x[1], reverse=True)
    top_two = [op for op, _ in sorted_op_history[:2]]
    banned_ops_line = (
        f"DO NOT USE these two most-frequent ineffective operations: {', '.join(top_two)}."
        if top_two
        else "No ineffective-operation history yet."
    )

    return (
        "NOTICE: Prefer simple operations: arithmetic combinations "
        "(+, -, *, /), log, sqrt, abs, or ratio between two features. "
        "Avoid aggregate or multi-step transformations. "
        + banned_ops_line
    )

def mask_frozen_features(frozen_features, frozen_list, useful_features, weak_features, top_features, original_feature_count=None):
    """
    frozen_features: dict, key is feature name, values is the last number of steps to freeze
    frozen_list: list of feature names that are currently frozen and should be updated.
    useful_features, weak_features, top_features: current feature groups
    original_feature_count: original number of features to enable protection mechanism
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 1. Reduce the freeze counters and remove expired features
    for key in list(frozen_features.keys()):
        frozen_features[key] -= 1
        if frozen_features[key] <= 0:
            del frozen_features[key]
    
    # 2. Add newly used features to the freeze dict
    for feat in frozen_list:
        if feat in frozen_features:
            # If the LLM hallucinated/reused an already frozen feature, 
            # just skip it to prevent the AssertionError.
            continue 
            
        if feat in top_features:
            frozen_features[feat] = FROZEN_STEPS["top"]
        elif feat in useful_features:
            frozen_features[feat] = FROZEN_STEPS["useful"]
        elif feat in weak_features:
            frozen_features[feat] = FROZEN_STEPS["weak"]
        else:
            # Fallback: still freeze unseen/misgrouped features to avoid immediate repetition loops.
            frozen_features[feat] = FROZEN_STEPS["weak"]
    
    # Clear the list now that we've processed the newly requested freezes
    frozen_list.clear()

    # 3. Explicitly remove ALL currently frozen features from the candidate groups
    top_features = [f for f in top_features if f not in frozen_features]
    useful_features = [f for f in useful_features if f not in frozen_features]
    weak_features = [f for f in weak_features if f not in frozen_features]

    # 4. Protection mechanism
    if original_feature_count is not None:
        total_available_features = len(set(useful_features + weak_features + top_features))
        half_original = original_feature_count / 2
        
        if total_available_features < half_original:
            logger.warning(f"Protection mechanism triggered: Available features ({total_available_features}) < half original ({half_original}). Recovering frozen features...")
            sorted_frozen = sorted(frozen_features.items(), key=lambda x: x[1])
            
            for feature_name, remaining_steps in sorted_frozen:
                if total_available_features >= half_original:
                    break
                
                if remaining_steps == FROZEN_STEPS["top"]:
                    top_features.append(feature_name)
                elif remaining_steps == FROZEN_STEPS["useful"]:
                    useful_features.append(feature_name)
                else:
                    weak_features.append(feature_name)
                
                del frozen_features[feature_name]
                total_available_features += 1
                logger.info(f"  Recovered feature '{feature_name}' (had {remaining_steps} freeze steps remaining)")
            
            logger.info(f"  Protection complete: Available features now = {total_available_features}")
    
    return useful_features, weak_features, top_features, frozen_features     
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logging(results_dir, output_dir):
    """
    Configure a run-specific logger that writes to console and file.
    """
    log_dir = os.path.join(results_dir, "logs", output_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run.log")

    logger = logging.getLogger('AutoFE')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def format_noise_tag(noise_level):
    return f"{noise_level:.2f}".replace('.', 'p')

def build_experiment_tag(ablation_mode, noise_level):
    if ablation_mode in {"ours", "noise_sensitivity", "random_grouping", "rank_only"}:
        return f"{ablation_mode}_noise_{format_noise_tag(noise_level)}"
    return ablation_mode

def resolve_llm_generation_config(llm_preset, temperature_override=None):
    if llm_preset == "debug":
        return {
            "temperature": 0.0,
            "top_p": None,
            "uses_sampling": False,
        }

    if llm_preset in {"coding", "math"}:
        return {
            "temperature": 0.7 if temperature_override is None else temperature_override,
            "top_p": 0.95,
            "uses_sampling": True,
        }

    raise ValueError(f"Unsupported llm_preset: {llm_preset}")


def encode_features_for_noise_analysis(x_train, x_val):
    """
    Encode non-numeric feature values for grouping/noise analysis only.
    The prompt-facing data keeps the original values so the LLM can still use
    semantic information from raw categorical fields.
    """
    x_train_encoded = x_train.copy()
    x_val_encoded = x_val.copy()

    cat_cols = x_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat_cols:
        return x_train_encoded, x_val_encoded

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    x_train_encoded[cat_cols] = encoder.fit_transform(x_train_encoded[cat_cols].astype(str))
    x_val_encoded[cat_cols] = encoder.transform(x_val_encoded[cat_cols].astype(str))

    return x_train_encoded, x_val_encoded


def get_grouping_inputs(x_train, x_val, masked):
    if masked:
        return x_train, x_val

    return encode_features_for_noise_analysis(x_train, x_val)

def save_run_config(results_dir, output_dir, config):
    config_dir = os.path.join(results_dir, "configs", output_dir)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "run_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def save_trace_records(trace_records, results_dir, output_dir):
    trace_dir = os.path.join(results_dir, "traces", output_dir)
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "ablation_trace.csv")

    pd.DataFrame(trace_records).to_csv(trace_path, index=False)

def evaluate_test_curve(x_train, y_train, x_val, y_val, x_test, y_test, task_model):
    x_train_final = pd.concat([x_train, x_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    _, test_results = eval_with_model(x_train_final, y_train_final, x_test, y_test, task_model)
    return test_results


def passes_acceptance_gate(diff_results, base_results, n_samples, n_classes=2):
    """
    samll samples → more noise → sticker gate
    statistic: standard error ∝ 1/sqrt(n)
    """
    confidence_factor = np.sqrt(2000 / n_samples)  # 2000 samples as the standard
    confidence_factor = np.clip(confidence_factor, 1.0, 3.0)  # max 3
    
    epsilon = base_results.get("acc_std", 0) / confidence_factor

    if n_classes > 2:
        delta_f1  = diff_results.get("f1",  -float("inf"))
        delta_acc = diff_results.get("acc", -float("inf"))
        return delta_f1 > 0 and delta_acc >= -epsilon
    else:
        delta_auc = diff_results.get("auc", -float("inf"))
        delta_acc = diff_results.get("acc", -float("inf"))
        return delta_auc > 0 and delta_acc >= -epsilon


def get_prompt_settings(ablation_mode="ours", noise_level=0.1):
    grouped_format = (
        "`{feature_name}: {importance_diff}, {unique_values/total_values}, "
        "{data_type}, {value_range}, {sample_values}`\n"
        "- `value_range`: (Max, Min)"
    )

    grouped_requirements = {
        "function_1_title": "Intra-Group Enhancement",
        "function_1_purpose": "Amplify existing signal patterns within the SAME displayed group.",
        "function_1_requirements": (
            "- Source: Features from the SAME displayed group.\n"
            "- Goal: Capture hidden patterns within high-signal or recoverable features."
        ),
        "function_2_title": "Cross-Group Synthesis",
        "function_2_purpose": "Unlock hidden interactions across DIFFERENT displayed groups.",
        "function_2_requirements": (
            "- Source: Combine features from DIFFERENT displayed groups.\n"
            "- Goal: Create synergistic combinations that exceed individual feature performance."
        ),
    }

    if ablation_mode in {"ours", "random_grouping", "noise_sensitivity"}:
        return {
            "grouping_description": (
                f"A noise column is used as the baseline anchor (noise level = {noise_level:.2f}). "
                "Features are ranked by SHAP and organized into Top, Useful, and Weak groups."
            ),
            "feature_format": (
                grouped_format + "\n"
                "- `importance_diff`: Difference from the noise baseline (`+` = higher, `-` = lower)."
            ),
            "info_mode": "noise_diff",
            "sections": [
                "Top Features (Top 10%, minimum 2 features)",
                "Moderately Useful Features",
                "Weak Features (performing at or below noise level)",
            ],
            **grouped_requirements,
        }
    raise ValueError(f"Unsupported ablation mode: {ablation_mode}")


def prepare_feature_groups(x_train, y_train, x_val, y_val, ablation_mode, noise_level, random_state, task_model):
    if ablation_mode in {"ours", "noise_sensitivity", "rank_only"}:
        importance_df, useful_features, weak_features, top_features, noise_threshold = divide_groups_by_shap(
            x_train, y_train, x_val, y_val, noise_level=noise_level, random_state=random_state, task_model="xgboost"
        )
    else:
        raise ValueError(f"Unsupported ablation mode: {ablation_mode}")

    return importance_df, useful_features, weak_features, top_features, noise_threshold


def build_prompt_payload(x_train, importance_df, top_features, useful_features, weak_features,
                         noise_threshold, ablation_mode, noise_level):
    prompt_settings = get_prompt_settings(ablation_mode, noise_level)
    info_mode = prompt_settings["info_mode"]

    if ablation_mode == "no_grouping":
        all_features_info = build_detailed_features_inform(
            x_train,
            None,
            x_train.columns.tolist(),
            None,
            info_mode=info_mode
        )
        feature_blocks = build_feature_blocks([
            (prompt_settings["sections"][0], all_features_info)
        ])
        top_features_info = all_features_info
        useful_features_info = "None"
        weak_features_info = "None"
    else:
        top_features_info = build_detailed_features_inform(
            x_train, importance_df, top_features, noise_threshold, info_mode=info_mode
        )
        useful_features_info = build_detailed_features_inform(
            x_train, importance_df, useful_features, noise_threshold, info_mode=info_mode
        )

        section_pairs = [
            (prompt_settings["sections"][0], top_features_info),
            (prompt_settings["sections"][1], useful_features_info),
        ]

        if USE_WEAK_GROUP_IN_PROMPT:
            weak_features_info = build_detailed_features_inform(
                x_train, importance_df, weak_features, noise_threshold, info_mode=info_mode
            )
            section_pairs.append((prompt_settings["sections"][2], weak_features_info))
        else:
            weak_features_info = "None"

        feature_blocks = build_feature_blocks(section_pairs)

    grouping_description = prompt_settings["grouping_description"]
    function_2_requirements = prompt_settings["function_2_requirements"]

    if not USE_WEAK_GROUP_IN_PROMPT and ablation_mode != "no_grouping":
        grouping_description += " Only the Top and Moderately Useful groups are shown in this prompt."
        function_2_requirements = (
            "- Source: Combine features from DIFFERENT displayed groups.\n"
            "- Priority: Combine complementary top/useful features when the interaction is justified.\n"
            "- Goal: Create synergistic combinations that exceed individual feature performance."
        )

    extra_replacements = {
        "GROUPING_DESCRIPTION": grouping_description,
        "FEATURE_FORMAT": prompt_settings["feature_format"],
        "FEATURE_BLOCKS": feature_blocks,
        "FUNCTION_1_TITLE": prompt_settings["function_1_title"],
        "FUNCTION_1_PURPOSE": prompt_settings["function_1_purpose"],
        "FUNCTION_1_REQUIREMENTS": prompt_settings["function_1_requirements"],
        "FUNCTION_2_TITLE": prompt_settings["function_2_title"],
        "FUNCTION_2_PURPOSE": prompt_settings["function_2_purpose"],
        "FUNCTION_2_REQUIREMENTS": function_2_requirements,
    }

    return top_features_info, useful_features_info, weak_features_info, extra_replacements


def append_summary_row(summary_path, summary_row):
    new_row = pd.DataFrame([summary_row])

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
        combined.to_csv(summary_path, index=False)
    else:
        new_row.to_csv(summary_path, index=False)


def main(dataset_path, output_dir, masked=True, steps=10,
         prompt_path=DEFAULT_PROMPT_PATH, port=8000, reject_weak=False,
         model='Qwen/Qwen2.5-7B-Instruct', results_dir="./results",
         ablation_mode="ours", noise_level=0.1, random_seed=42, temperature=None,
         track_test_curve=True, llm_preset="coding", task_model='xgboost', parallel_eval=False, parallel_cv=False):

    if ablation_mode not in ABLATION_MODES:
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
    if llm_preset not in LLM_PRESETS:
        raise ValueError(f"Unsupported llm_preset: {llm_preset}")

    llm_config = resolve_llm_generation_config(llm_preset, temperature)
    llm_temperature = llm_config["temperature"]
    logger = setup_logging(results_dir, output_dir)

    logger.info("========== Starting Experiment ==========")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(
        f"Ablation: {ablation_mode} | Noise Level: {noise_level:.2f} | "
        f"LLM Preset: {llm_preset} | Temperature: {llm_temperature:.2f} | Seed: {random_seed}"
    )

    save_run_config(results_dir, output_dir, {
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "masked": masked,
        "steps": steps,
        "prompt_path": prompt_path,
        "port": port,
        "reject_weak": reject_weak,
        "model": model,
        "results_dir": results_dir,
        "ablation_mode": ablation_mode,
        "noise_level": noise_level,
        "random_seed": random_seed,
        "llm_preset": llm_preset,
        "temperature": llm_temperature,
        "top_p": llm_config["top_p"],
        "uses_sampling": llm_config["uses_sampling"],
        "track_test_curve": track_test_curve,
    })

    try:
        logger.info("Loading datasets...")
        train_ds, val_ds, test_ds = load_datasets(dataset_path, masked=masked)

        logger.info("Masking feature names...")
        train_ds = mask_feature_names(train_ds)
        val_ds = mask_feature_names(val_ds)
        test_ds = mask_feature_names(test_ds)
        if not masked:
            logger.info(
                "Raw dataset mode detected: keeping original feature values for prompting "
                "and encoding temporary copies only for grouping/noise analysis."
            )

        x_train, y_train = get_X_y(train_ds)
        x_val, y_val = get_X_y(val_ds)
        x_test, y_test = get_X_y(test_ds)

        n_classes = len(np.unique(y_train))
        n_samples = train_ds.shape[0] + val_ds.shape[0]
        base_feature_count = x_train.shape[1]

        main_score_type = 'auc' if n_classes == 2 else 'f1'

        logger.info(f"Data shape: Train {x_train.shape}, Val {x_val.shape}, Test {x_test.shape}")
        logger.info(f"Model: {model} | Steps: {steps} | Main Score Type: {main_score_type}")

        logger.info("Evaluating baseline model...")
        # _, current_results = eval_with_xgb(x_train, y_train, x_val, y_val)
        _, current_results = eval_with_model(x_train, y_train, x_val, y_val, task_model)
        logger.info(f"Baseline Results: {current_results}")

        winner_col_name = ''
        winner_type = ''
        pre_importance_df = None
        pre_useful_features = []
        pre_weak_features = []
        pre_top_features = []
        pre_noise_threshold = None
        pre_x_train = x_train.copy()
        pre_x_val = x_val.copy()
        pre_x_test = x_test.copy()
        pre_results = current_results.copy()
        current_step = 0
        previous_results = {}
        generated_features = []
        rejected_features = []
        trace_records = []


        # frozen chosen features for certain steps to avoid duplicate generation
        frozen_features = {}
        frozen_list = []

        failed_steps = 0
        op_history = {}

        baseline_test_results = None
        if track_test_curve:
            baseline_test_results = evaluate_test_curve(x_train, y_train, x_val, y_val, x_test, y_test, task_model)

        trace_records.append({
            "step": 0,
            "decision": "baseline",
            "accepted_feature": "",
            "accepted_type": "",
            "feature_count": x_train.shape[1],
            "ablation_mode": ablation_mode,
            "llm_preset": llm_preset,
            "noise_level": noise_level,
            "temperature": llm_temperature,
            "top_p": llm_config["top_p"] if llm_config["top_p"] is not None else np.nan,
            "top_group_size": np.nan,
            "useful_group_size": np.nan,
            "weak_group_size": np.nan,
            "intra_feature": "",
            "intra_success": False,
            "intra_auc_imp": np.nan,
            "intra_acc_imp": np.nan,
            "intra_score_diff": np.nan,
            "cross_feature": "",
            "cross_success": False,
            "cross_auc_imp": np.nan,
            "cross_acc_imp": np.nan,
            "cross_score_diff": np.nan,
            "val_acc": current_results["acc"],
            "val_auc": current_results["auc"],
            "val_f1": current_results["f1"],
            "test_acc": baseline_test_results["acc"] if baseline_test_results else np.nan,
            "test_auc": baseline_test_results["auc"] if baseline_test_results else np.nan,
            "test_f1": baseline_test_results["f1"] if baseline_test_results else np.nan,
            "frozen_features": str(list(frozen_features.keys())),
        })
        save_trace_records(trace_records, results_dir, output_dir)

    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None

    while current_step < steps and failed_steps < FORCE_END:
        logger.info(f"\n>>> [Step {current_step + 1}/{steps}] Processing...")

        try:
            step_seed = random_seed + current_step
            grouping_x_train, grouping_x_val = get_grouping_inputs(x_train, x_val, masked)
            importance_df, useful_features, weak_features, top_features, noise_threshold = prepare_feature_groups(
                grouping_x_train, y_train, grouping_x_val, y_val,
                ablation_mode=ablation_mode,
                noise_level=noise_level,
                random_state=step_seed,
                task_model=task_model
            )
            
            # if there are no accepted features after continuously K steps
            # we release all the frozen features to reset the state
            if failed_steps >= K_PATIENCE:
                failed_steps = 0
                frozen_features = {}
                frozen_list = []
                logger.info("Patience reached: reset frozen state and pending freeze list.")
            
            operations_info = operations_prompt_info(op_history)

            useful_features, weak_features, top_features, frozen_features = mask_frozen_features(
                frozen_features,
                frozen_list,
                useful_features,
                weak_features,
                top_features,
                original_feature_count=base_feature_count,
            )

            if reject_weak and ablation_mode in WEAK_MONITOR_MODES:
                if (
                    winner_col_name and
                    winner_col_name in weak_features and
                    winner_type in previous_results
                ):
                    logger.warning(
                        f"Overfitting detected: '{winner_col_name}' moved to weak group. Rolling back..."
                    )

                    importance_df = pre_importance_df
                    useful_features = pre_useful_features[:]
                    weak_features = pre_weak_features[:]
                    top_features = pre_top_features[:]
                    noise_threshold = pre_noise_threshold
                    x_train = pre_x_train.copy()
                    x_val = pre_x_val.copy()
                    x_test = pre_x_test.copy()
                    current_results = pre_results.copy()

                    previous_results[winner_type]["status"] = "rolled_back"
                    rejected_features.append(winner_col_name)

                    if generated_features and generated_features[-1]["col_name"] == winner_col_name:
                        generated_features.pop()
                else:
                    pre_importance_df = importance_df
                    pre_useful_features = useful_features[:]
                    pre_weak_features = weak_features[:]
                    pre_top_features = top_features[:]
                    pre_noise_threshold = noise_threshold
                    pre_x_train = x_train.copy()
                    pre_x_val = x_val.copy()
                    pre_x_test = x_test.copy()
                    pre_results = current_results.copy()
            elif reject_weak and ablation_mode not in WEAK_MONITOR_MODES:
                logger.info("reject_weak is ignored for ablation modes without stable weak-group semantics.")

            top_features_info, useful_features_info, weak_features_info, extra_replacements = build_prompt_payload(
                x_train=x_train,
                importance_df=importance_df,
                top_features=top_features,
                useful_features=useful_features,
                weak_features=weak_features,
                noise_threshold=noise_threshold,
                ablation_mode=ablation_mode,
                noise_level=noise_level,
            )

            previous_report = "None" if current_step == 0 else build_previous_report(previous_results)
            top_5_strategies = "None" if current_step == 0 else build_top_k_report(generated_features)

            clean_features = [str(x) for x in rejected_features if x is not None] if rejected_features else []
            rejected_features_info = ", ".join(clean_features) if clean_features else "None"
        

            prompt = fill_prompt_template(
                cls_model=task_model,
                n_classes=n_classes,
                imp_type='SHAP',
                top_features=top_features_info,
                useful_features=useful_features_info,
                weak_features=weak_features_info,
                score="None",
                previous_report=previous_report,
                top_5=top_5_strategies,
                reject_features=rejected_features_info,
                prompt_path=prompt_path,
                extra_replacements=extra_replacements,
                operations_info=operations_info,
            )

            logger.info("Calling LLM...")
            answer = call_llm(
                prompt=prompt,
                model_name=model,
                port=port,
                temperature=llm_temperature,
                top_p=llm_config["top_p"],
            )
            save_prompt_answer(
                prompt=prompt,
                answer=answer,
                step=current_step,
                results_dir=results_dir,
                output_dir=output_dir,
            )

            code = extract_code_block(answer)
            save_code(code=code, results_dir=results_dir, step=current_step, output_dir=output_dir)

            logger.info("Executing generated code...")
            success_flag, train_data = apply_code(code, x_train)
            _, val_data = apply_code(code, x_val)
            _, test_data = apply_code(code, x_test)

            intra_feat_name = train_data["intra"]["col_name"]
            cross_feat_name = train_data["cross"]["col_name"]

            logger.info(f"Proposed Features: Intra='{intra_feat_name}', Cross='{cross_feat_name}'")

            intra_used_feats, intra_name_ops = parse_generated_features(intra_feat_name)
            cross_used_feats, cross_name_ops = parse_generated_features(cross_feat_name)
            intra_used_ops = sorted(set(intra_name_ops))
            cross_used_ops = sorted(set(cross_name_ops))

            if intra_feat_name:
                frozen_list += intra_used_feats
            if cross_feat_name:
                frozen_list += cross_used_feats

            # Fallback: parse features from generated code in case col_name is not strictly formatted.
            frozen_list += parse_features_from_code(code)
            frozen_list = list(set(frozen_list))

            active_frozen = set(frozen_features.keys())
            intra_hits_frozen = sorted(active_frozen.intersection(intra_used_feats))
            cross_hits_frozen = sorted(active_frozen.intersection(cross_used_feats))

            if intra_hits_frozen:
                logger.warning(
                    f"Intra feature '{intra_feat_name}' uses frozen base features {intra_hits_frozen}; "
                    "force-rejecting this candidate."
                )
                train_data["intra"]["success"] = False
                val_data["intra"]["success"] = False
                test_data["intra"]["success"] = False
            if cross_hits_frozen:
                logger.warning(
                    f"Cross feature '{cross_feat_name}' uses frozen base features {cross_hits_frozen}; "
                    "force-rejecting this candidate."
                )
                train_data["cross"]["success"] = False
                val_data["cross"]["success"] = False
                test_data["cross"]["success"] = False

            step_record = {
                "step": current_step + 1,
                "ablation_mode": ablation_mode,
                "llm_preset": llm_preset,
                "noise_level": noise_level,
                "temperature": llm_temperature,
                "top_p": llm_config["top_p"] if llm_config["top_p"] is not None else np.nan,
                "feature_count": x_train.shape[1],
                "top_group_size": len(top_features),
                "useful_group_size": len(useful_features),
                "weak_group_size": len(weak_features),
                "intra_feature": intra_feat_name or "",
                "intra_success": False,
                "intra_auc_imp": np.nan,
                "intra_acc_imp": np.nan,
                "intra_score_diff": np.nan,
                "cross_feature": cross_feat_name or "",
                "cross_success": False,
                "cross_auc_imp": np.nan,
                "cross_acc_imp": np.nan,
                "cross_score_diff": np.nan,
                "decision": "error",
                "accepted_feature": "",
                "accepted_type": "",
            }

            if not success_flag:
                logger.warning(f"Step {current_step}: Code execution failed. Rejecting both.")
                rejected_features.extend([name for name in [intra_feat_name, cross_feat_name] if name])
                previous_results["intra"] = {"type": "intra", "col_name": intra_feat_name, "status": "error"}
                previous_results["cross"] = {"type": "cross", "col_name": cross_feat_name, "status": "error"}
                step_record["decision"] = "rejected"
                step_record["val_acc"] = current_results["acc"]
                step_record["val_auc"] = current_results["auc"]
                step_record["val_f1"] = current_results["f1"]
                current_test_results = (
                    evaluate_test_curve(x_train, y_train, x_val, y_val, x_test, y_test, task_model)
                    if track_test_curve else None
                )
                step_record["test_acc"] = current_test_results["acc"] if current_test_results else np.nan
                step_record["test_auc"] = current_test_results["auc"] if current_test_results else np.nan
                step_record["test_f1"] = current_test_results["f1"] if current_test_results else np.nan
                trace_records.append(step_record)
                save_trace_records(trace_records, results_dir, output_dir)
                current_step += 1
                failed_steps += 1
                continue

            score_i = -float('inf')
            score_c = -float('inf')
            res_i = None
            res_c = None
            x_tr_i = x_val_i = x_te_i = None
            x_tr_c = x_val_c = x_te_c = None
            gate_i = False
            gate_c = False

            intra_all_success = (
                train_data["intra"]["success"] and
                val_data["intra"]["success"] and
                test_data["intra"]["success"]
            )
            cross_all_success = (
                train_data["cross"]["success"] and
                val_data["cross"]["success"] and
                test_data["cross"]["success"]
            )

            # if intra_all_success:
            #     x_tr_i, x_val_i, x_te_i = (
            #         train_data["intra"]["df"],
            #         val_data["intra"]["df"],
            #         test_data["intra"]["df"],
            #     )
            #     # _, res_i = eval_with_xgb(x_tr_i, y_train, x_val_i, y_val)
            #     _, res_i = eval_with_model(x_tr_i, y_train, x_val_i, y_val, task_model)
            #     diff_i = eval_new_generated_features(res_i, current_results)
            #     gate_i = passes_acceptance_gate(diff_i, res_i, n_samples=n_samples, n_classes=n_classes)
            #     score_i = diff_i[main_score_type]
            #     step_record["intra_success"] = True
            #     step_record["intra_auc_imp"] = diff_i["auc"]
            #     step_record["intra_acc_imp"] = diff_i["acc"]
            #     step_record["intra_score_diff"] = score_i
            #     logger.info(
            #         f"Intra Eval: '{intra_feat_name}' -> "
            #         f"{main_score_type.upper()} Δ {score_i:.4f}| "
            #         f"ACC Δ {diff_i['acc']:.4f} | Gate={gate_i}"
            #     )

            #     previous_results["intra"] = {
            #         "type": train_data["intra"]["type"],
            #         "col_name": intra_feat_name,
            #         "score_diff": score_i,
            #         "auc_imp": diff_i["auc"],
            #         "acc_imp": diff_i["acc"],
            #         "status": "passed_gate" if gate_i else "failed_gate",
            #     }
            # else:
            #     logger.warning(f"Intra Feature '{intra_feat_name}' generation failed/empty on at least one split.")
            #     previous_results["intra"] = {"type": "intra", "col_name": intra_feat_name, "status": "error"}

            # if cross_all_success:
            #     x_tr_c, x_val_c, x_te_c = (
            #         train_data["cross"]["df"],
            #         val_data["cross"]["df"],
            #         test_data["cross"]["df"],
            #     )
            #     # _, res_c = eval_with_xgb(x_tr_c, y_train, x_val_c, y_val)
            #     _, res_c = eval_with_model(x_tr_c, y_train, x_val_c, y_val, task_model)
            #     diff_c = eval_new_generated_features(res_c, current_results)
            #     gate_c = passes_acceptance_gate(diff_c, res_c, n_samples=n_samples, n_classes=n_classes)
            #     score_c = diff_c[main_score_type]
            #     step_record["cross_success"] = True
            #     step_record["cross_auc_imp"] = diff_c["auc"]
            #     step_record["cross_acc_imp"] = diff_c["acc"]
            #     step_record["cross_score_diff"] = score_c
            #     logger.info(
            #         f"Cross Eval: '{cross_feat_name}' -> "
            #         f"{main_score_type.upper()} Δ {score_c:.4f}| "
            #         f"ACC Δ {diff_c['acc']:.4f} | Gate={gate_c}"
            #     )

            #     previous_results["cross"] = {
            #         "type": train_data["cross"]["type"],
            #         "col_name": cross_feat_name,
            #         "score_diff": score_c,
            #         "auc_imp": diff_c["auc"],
            #         "acc_imp": diff_c["acc"],
            #         "status": "passed_gate" if gate_c else "failed_gate",
            #     }
            # else:
            #     logger.warning(f"Cross Feature '{cross_feat_name}' generation failed/empty on at least one split.")
            #     previous_results["cross"] = {"type": "cross", "col_name": cross_feat_name, "status": "error"}
            # ── 先取出各自的 df（保持原有 success 判断） ────────────────
            if intra_all_success:
                x_tr_i, x_val_i, x_te_i = (
                    train_data["intra"]["df"],
                    val_data["intra"]["df"],
                    test_data["intra"]["df"],
                )
            if cross_all_success:
                x_tr_c, x_val_c, x_te_c = (
                    train_data["cross"]["df"],
                    val_data["cross"]["df"],
                    test_data["cross"]["df"],
                )
            
            # ── 构造任务列表 ─────────────────────────────────────────────
            eval_tasks = []
            if intra_all_success:
                eval_tasks.append(('intra', x_tr_i, x_val_i, x_te_i))
            if cross_all_success:
                eval_tasks.append(('cross', x_tr_c, x_val_c, x_te_c))
            
            eval_worker_args = [
                (tag, x_tr, x_va, x_te, y_train, y_val, task_model, parallel_cv, step_seed)
                for tag, x_tr, x_va, x_te in eval_tasks
            ]
            
            # ── 执行评估 ─────────────────────────────────────────────────
            eval_results = {}   # tag -> (clf, res)
            
            if parallel_eval and len(eval_worker_args) > 1:
                # spawn：intra 和 cross 各自在子进程里跑，各自独占 CUDA context
                # 显存会同时上涨（两个 XGBoost GPU / MLP 同时在卡上）
                spawn_ctx = tmp.get_context('spawn')
                with spawn_ctx.Pool(processes=len(eval_worker_args)) as pool:
                    for tag, clf, res in pool.map(_eval_feature_worker, eval_worker_args):
                        eval_results[tag] = (clf, res)
            else:
                # 串行（默认）
                for args in eval_worker_args:
                    tag, clf, res = _eval_feature_worker(args)
                    eval_results[tag] = (clf, res)
            
            # ── 解包 intra ───────────────────────────────────────────────
            if intra_all_success:
                if eval_results.get('intra') is not None:
                    _, res_i = eval_results['intra']
                    diff_i   = eval_new_generated_features(res_i, current_results)
                    gate_i   = passes_acceptance_gate(diff_i, res_i, n_samples=n_samples, n_classes=n_classes)
                    score_i  = diff_i[main_score_type]
                    step_record["intra_success"]    = True
                    step_record["intra_auc_imp"]    = diff_i["auc"]
                    step_record["intra_acc_imp"]    = diff_i["acc"]
                    step_record["intra_score_diff"] = score_i
                    logger.info(
                        f"Intra Eval: '{intra_feat_name}' -> "
                        f"{main_score_type.upper()} Δ {score_i:.4f} | "
                        f"ACC Δ {diff_i['acc']:.4f} | Gate={gate_i}"
                    )
                    previous_results["intra"] = {
                        "type":       train_data["intra"]["type"],
                        "col_name":   intra_feat_name,
                        "score_diff": score_i,
                        "auc_imp":    diff_i["auc"],
                        "acc_imp":    diff_i["acc"],
                        "status":     "passed_gate" if gate_i else "failed_gate",
                    }
                else:
                    intra_all_success = False
            
            if not intra_all_success:
                logger.warning(f"Intra Feature '{intra_feat_name}' generation/eval failed.")
                previous_results["intra"] = {"type": "intra", "col_name": intra_feat_name, "status": "error"}
            
            # ── 解包 cross ───────────────────────────────────────────────
            if cross_all_success:
                if eval_results.get('cross') is not None:
                    _, res_c = eval_results['cross']
                    diff_c   = eval_new_generated_features(res_c, current_results)
                    gate_c   = passes_acceptance_gate(diff_c, res_c, n_samples=n_samples, n_classes=n_classes)
                    score_c  = diff_c[main_score_type]
                    step_record["cross_success"]    = True
                    step_record["cross_auc_imp"]    = diff_c["auc"]
                    step_record["cross_acc_imp"]    = diff_c["acc"]
                    step_record["cross_score_diff"] = score_c
                    logger.info(
                        f"Cross Eval: '{cross_feat_name}' -> "
                        f"{main_score_type.upper()} Δ {score_c:.4f} | "
                        f"ACC Δ {diff_c['acc']:.4f} | Gate={gate_c}"
                    )
                    previous_results["cross"] = {
                        "type":       train_data["cross"]["type"],
                        "col_name":   cross_feat_name,
                        "score_diff": score_c,
                        "auc_imp":    diff_c["auc"],
                        "acc_imp":    diff_c["acc"],
                        "status":     "passed_gate" if gate_c else "failed_gate",
                    }
                else:
                    cross_all_success = False
            
            if not cross_all_success:
                logger.warning(f"Cross Feature '{cross_feat_name}' generation/eval failed.")
                previous_results["cross"] = {"type": "cross", "col_name": cross_feat_name, "status": "error"}


            # Count ineffective operations only: if candidate does not improve the main score.
            intra_improved = intra_all_success and score_i > 0
            cross_improved = cross_all_success and score_c > 0

            if intra_feat_name and not intra_improved:
                for op in intra_used_ops:
                    op_history[op] = op_history.get(op, 0) + 1
            if cross_feat_name and not cross_improved:
                for op in cross_used_ops:
                    op_history[op] = op_history.get(op, 0) + 1

            if not gate_c and not gate_i:
                logger.info("DECISION: Both features rejected.")
                rejected_features.extend([name for name in [intra_feat_name, cross_feat_name] if name])
                winner_col_name, winner_type = "", ""
                step_record["decision"] = "rejected"
                if "intra" in previous_results and previous_results["intra"].get("status") == "passed_gate":
                    previous_results["intra"]["status"] = "rejected"
                if "cross" in previous_results and previous_results["cross"].get("status") == "passed_gate":
                    previous_results["cross"]["status"] = "rejected"
                failed_steps += 1
            elif gate_c and (not gate_i or score_c > score_i):
                logger.info(
                    f"DECISION: Accepted Cross Feature '{cross_feat_name}' "
                    f"(AUC Δ {step_record['cross_auc_imp']:.4f}, ACC Δ {step_record['cross_acc_imp']:.4f})"
                )
                winner_col_name = cross_feat_name
                winner_type = "cross"
                previous_results["cross"]["status"] = "accepted"
                if not gate_i and intra_feat_name:
                    rejected_features.append(intra_feat_name)
                elif "intra" in previous_results:
                    previous_results["intra"]["status"] = "rejected"

                generated_features.append(previous_results["cross"])
                x_train, x_val, x_test = x_tr_c, x_val_c, x_te_c
                
                x_train = mask_feature_names(x_train)
                x_val = mask_feature_names(x_val)
                x_test = mask_feature_names(x_test)

                current_results = res_c
                step_record["decision"] = "accepted_cross"
                step_record["accepted_feature"] = cross_feat_name or ""
                step_record["accepted_type"] = previous_results["cross"]["type"]
                failed_steps = 0  # reset failed steps on success
            else:
                logger.info(
                    f"DECISION: Accepted Intra Feature '{intra_feat_name}' "
                    f"(AUC Δ {step_record['intra_auc_imp']:.4f}, ACC Δ {step_record['intra_acc_imp']:.4f})"
                )
                winner_col_name = intra_feat_name
                winner_type = "intra"
                previous_results["intra"]["status"] = "accepted"
                if not gate_c and cross_feat_name:
                    rejected_features.append(cross_feat_name)
                elif "cross" in previous_results:
                    previous_results["cross"]["status"] = "rejected"

                generated_features.append(previous_results["intra"])
                x_train, x_val, x_test = x_tr_i, x_val_i, x_te_i

                x_train = mask_feature_names(x_train)
                x_val = mask_feature_names(x_val)
                x_test = mask_feature_names(x_test)

                current_results = res_i
                step_record["decision"] = "accepted_intra"
                step_record["accepted_feature"] = intra_feat_name or ""
                step_record["accepted_type"] = previous_results["intra"]["type"]
                failed_steps = 0  # reset failed steps on success

            current_test_results = (
                evaluate_test_curve(x_train, y_train, x_val, y_val, x_test, y_test, task_model)
                if track_test_curve else None
            )

            step_record["feature_count"] = x_train.shape[1]
            step_record["val_acc"] = current_results["acc"]
            step_record["val_auc"] = current_results["auc"]
            step_record["val_f1"] = current_results["f1"]
            step_record["test_acc"] = current_test_results["acc"] if current_test_results else np.nan
            step_record["test_auc"] = current_test_results["auc"] if current_test_results else np.nan
            step_record["test_f1"] = current_test_results["f1"] if current_test_results else np.nan
            trace_records.append(step_record)
            save_trace_records(trace_records, results_dir, output_dir)

            logger.info(f"Current Best {main_score_type.upper()}: {current_results[main_score_type]:.4f}")
            current_step += 1

        except Exception as e:
            logger.error(f"Step {current_step}: Unexpected Error - {e}")
            logger.error(traceback.format_exc())

            trace_records.append({
                "step": current_step + 1,
                "decision": "error",
                "accepted_feature": "",
                "accepted_type": "",
                "feature_count": x_train.shape[1],
                "ablation_mode": ablation_mode,
                "llm_preset": llm_preset,
                "noise_level": noise_level,
                "temperature": llm_temperature,
                "top_p": llm_config["top_p"] if llm_config["top_p"] is not None else np.nan,
                "top_group_size": np.nan,
                "useful_group_size": np.nan,
                "weak_group_size": np.nan,
                "intra_feature": "",
                "intra_success": False,
                "intra_auc_imp": np.nan,
                "intra_acc_imp": np.nan,
                "intra_score_diff": np.nan,
                "cross_feature": "",
                "cross_success": False,
                "cross_auc_imp": np.nan,
                "cross_acc_imp": np.nan,
                "cross_score_diff": np.nan,
                "val_acc": current_results["acc"],
                "val_auc": current_results["auc"],
                "val_f1": current_results["f1"],
                "test_acc": np.nan,
                "test_auc": np.nan,
                "test_f1": np.nan,
            })
            save_trace_records(trace_records, results_dir, output_dir)
            current_step += 1
            continue

    if reject_weak and ablation_mode in WEAK_MONITOR_MODES:
        grouping_x_train, grouping_x_val = get_grouping_inputs(x_train, x_val, masked)
        importance_df, useful_features, weak_features, top_features, noise_threshold = prepare_feature_groups(
            grouping_x_train, y_train, grouping_x_val, y_val,
            ablation_mode=ablation_mode,
            noise_level=noise_level,
            random_state=random_seed + current_step,
            task_model=task_model
        )
        if (
            winner_col_name and
            winner_col_name in weak_features and
            winner_type in previous_results
        ):
            logger.warning(f"Overfitting detected: '{winner_col_name}' moved to weak group. Rolling back...")
            x_train = pre_x_train.copy()
            x_val = pre_x_val.copy()
            x_test = pre_x_test.copy()

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Feature Engineering using LLM')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--masked', type=str2bool, default=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--reject_weak', type=str2bool, default=False)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--prompt_path', type=str, default=DEFAULT_PROMPT_PATH)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--results_dir', type=str, default="./results")
    parser.add_argument('--ablation_mode', type=str, default='ours', choices=sorted(ABLATION_MODES))
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--track_test_curve', type=str2bool, default=True)
    parser.add_argument('--llm_preset', type=str, default='coding', choices=sorted(LLM_PRESETS))
    parser.add_argument('--task_model', type=str, default='xgboost')

    args = parser.parse_args()

    dataset_name = args.dataset_path.strip('/').split('/')[-2]
    seed = args.dataset_path.strip('/').split('/')[-1]
    experiment_tag = build_experiment_tag(args.ablation_mode, args.noise_level)
    output_dir = os.path.join(experiment_tag, dataset_name, seed)

    x_train, y_train, x_val, y_val, x_test, y_test = main(
        dataset_path=args.dataset_path,
        masked=args.masked,
        steps=args.steps,
        prompt_path=args.prompt_path,
        model=args.model,
        results_dir=args.results_dir,
        reject_weak=args.reject_weak,
        port=args.port,
        output_dir=output_dir,
        ablation_mode=args.ablation_mode,
        noise_level=args.noise_level,
        random_seed=args.random_seed,
        temperature=args.temperature,
        track_test_curve=args.track_test_curve,
        llm_preset=args.llm_preset,
        task_model=args.task_model,
    )

    if x_train is not None:
        try:
            final_llm_config = resolve_llm_generation_config(args.llm_preset, args.temperature)
            x_train_final = pd.concat([x_train, x_val], axis=0)
            y_train_final = pd.concat([y_train, y_val], axis=0)

            logger = logging.getLogger('AutoFE')
            logger.info("\n========== Final Evaluation on Test Set ==========")

            # _, results = eval_with_xgb_final(x_train_final, y_train_final, x_test, y_test)
            _, results = eval_with_model_final(x_train_final, y_train_final, x_test, y_test, args.task_model)
            summary_path = os.path.join(args.results_dir, 'summary.csv')

            append_summary_row(summary_path, {
                'dataset': dataset_name,
                'seed': seed,
                'ablation_mode': args.ablation_mode,
                'noise_level': args.noise_level,
                'llm_preset': args.llm_preset,
                'temperature': final_llm_config['temperature'],
                'top_p': final_llm_config['top_p'],
                'prompt_path': args.prompt_path,
                'feature_count': x_train.shape[1],
                'acc': results["acc"],
                'auc': results['auc'],
                'f1': results['f1'],
            })

            logger.info(f"Test Accuracy: {results['acc']:.4f}")
            logger.info(f"Test AUC:      {results['auc']:.4f}")
            logger.info(f"Test F1:       {results['f1']:.4f}")
            logger.info(f"Results saved to {summary_path}")
            logger.info("========== Experiment Finished ==========")

        except Exception as e:
            print(f"Final Evaluation Failed: {e}")
            traceback.print_exc()
