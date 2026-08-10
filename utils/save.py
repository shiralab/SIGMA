import os
import json
import pickle
from datetime import datetime

import os

def save_prompt_answer(prompt, answer, results_dir, step, output_dir):
    """
    将 Prompt 和 LLM 的回答合并保存到一个 step_{step}.txt 文件中。
    """
    save_dir = os.path.join(results_dir, 'prompts')
    save_dir = os.path.join(save_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"step_{step}.txt")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*30 + f" STEP {step} PROMPT " + "="*30 + "\n")
        f.write(prompt)
        f.write("\n\n" + "="*30 + f" STEP {step} ANSWER " + "="*30 + "\n")
        f.write(answer)
        f.write("\n" + "="*75 + "\n")
    
    print(f"[SAVED] Step {step} combined file saved to {file_path}")


def save_strategy_result(strategy_record, results_dir, step):
    """
    Save detailed strategy record for successful feature engineering step.
    
    Args:
        strategy_record (dict): Contains step, new_features, types, improvement, results, code
        results_dir (str): Root directory to save results
        step (int): Current optimization step number
    """
    save_dir = os.path.join(results_dir, 'strategies')
    os.makedirs(save_dir, exist_ok=True)
    
    strategy_path = os.path.join(save_dir, f"step_{step}_strategy.json")
    
    # Convert strategy record to JSON-serializable format
    strategy_json = {
        'step': strategy_record['step'],
        'new_features': strategy_record['new_features'],
        'feature_types': strategy_record['feature_types'],
        'improvement': float(strategy_record['improvement']),
        'results': {k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in strategy_record['results'].items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump(strategy_json, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] Step {step} strategy saved to {strategy_path}")


def save_optimization_history(past_strategies, rejected_features, generated_features, results_dir):
    """
    Save complete optimization history including top strategies and rejected features.
    
    Args:
        past_strategies (dict): Dictionary containing 'top_5' and other historical data
        rejected_features (list): List of feature names that were rejected
        generated_features (list): List of all generated feature names
        results_dir (str): Root directory to save results
    """
    history_dir = os.path.join(results_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    
    history = {
        'timestamp': datetime.now().isoformat(),
        'total_generated_features': len(generated_features),
        'total_rejected_features': len(rejected_features),
        'generated_features': generated_features,
        'rejected_features': rejected_features,
        'top_5_strategies': past_strategies.get('top_5', [])
    }
    
    history_path = os.path.join(history_dir, 'optimization_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] Optimization history saved to {history_path}")


def save_final_results(results, results_dir):
    """
    Save final optimization results including model performance and metrics.
    
    Args:
        results (dict): Dictionary containing:
            - model: trained XGBoost model
            - test_results: test set performance metrics
            - generated_features: list of generated features
            - rejected_features: list of rejected features
            - best_results: best validation results
            - past_strategies: historical strategy information
        results_dir (str): Root directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Save test results as JSON
    test_results_path = os.path.join(results_dir, 'final_test_results.json')
    test_results = {
        'accuracy': float(results['test_results']['acc']),
        'auc': float(results['test_results']['auc']),
        'f1': float(results['test_results']['f1']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # Save model using pickle
    model_path = os.path.join(results_dir, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(results['model'], f)
    
    # Save summary statistics
    summary_path = os.path.join(results_dir, 'summary.json')
    summary = {
        'total_features_generated': len(results['generated_features']),
        'total_features_retained': len(results['generated_features']) - len(results['rejected_features']),
        'total_features_rejected': len(results['rejected_features']),
        'test_accuracy': float(results['test_results']['acc']),
        'test_auc': float(results['test_results']['auc']),
        'test_f1': float(results['test_results']['f1']),
        'best_validation_results': {k: float(v) if isinstance(v, (int, float)) else v 
                                    for k, v in results['best_results'].items()},
        'generated_features': results['generated_features'],
        'rejected_features': results['rejected_features'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save final data as CSV
    if 'x_train_final' in results:
        train_path = os.path.join(results_dir, 'x_train_with_engineered_features.csv')
        results['x_train_final'].to_csv(train_path, index=False)
    
    if 'x_test_final' in results:
        test_path = os.path.join(results_dir, 'x_test_with_engineered_features.csv')
        results['x_test_final'].to_csv(test_path, index=False)
    
    print(f"[SAVED] Final results saved to {results_dir}")
    print(f"  - Test results: {test_results_path}")
    print(f"  - Model: {model_path}")
    print(f"  - Summary: {summary_path}")


def save_iteration_log(step, importance_df, top_features, useful_features, weak_features, 
                       current_results, generated_features, improvement, results_dir):
    """
    Save detailed log for each optimization iteration.
    
    Args:
        step (int): Current optimization step
        importance_df (pd.DataFrame): Feature importance scores
        top_features (list): Top-ranked features
        useful_features (list): Useful features
        weak_features (list): Weak features
        current_results (dict): Current model performance metrics
        generated_features (list): Features generated in this step
        improvement (float): Improvement score from this step
        results_dir (str): Root directory to save results
    """
    log_dir = os.path.join(results_dir, 'iteration_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"step_{step:03d}_log.json")
    
    # Save importance DataFrame
    importance_df.to_csv(os.path.join(log_dir, f"step_{step:03d}_importance.csv"), index=False)
    
    iteration_log = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'features': {
            'top_features': top_features,
            'useful_features': useful_features,
            'weak_features': weak_features,
            'generated_features': generated_features
        },
        'model_results': {k: float(v) if isinstance(v, (int, float)) else v 
                         for k, v in current_results.items()},
        'improvement': float(improvement)
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(iteration_log, f, indent=2, ensure_ascii=False)


def load_results(results_dir):
    """
    Load saved optimization results from directory.
    
    Args:
        results_dir (str): Root directory containing saved results
    
    Returns:
        dict: Contains loaded model, results, and metadata
    """
    results = {}
    
    # Load model
    model_path = os.path.join(results_dir, 'final_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            results['model'] = pickle.load(f)
    
    # Load summary
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            results['summary'] = json.load(f)
    
    # Load test results
    test_results_path = os.path.join(results_dir, 'final_test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results['test_results'] = json.load(f)
    
    print(f"[LOADED] Results loaded from {results_dir}")
    return results
