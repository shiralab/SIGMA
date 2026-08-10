
def eval_new_generated_features(new_results, ori_results):
    diff_results = {}
    for metric, score in ori_results.items():
        diff_results[metric] = new_results[metric] - ori_results[metric]
    
    return diff_results

def constrained_optim_gate(diff_results, base_results, n_classes=2):
    """
    Constrained Optimization:
    maximize  Δauc (or Δf1 for multiclass)
    subject to  Δacc ≥ -ε
    
    ε = acc_std of baseline
    """
    epsilon = base_results.get("acc_std", 0)

    if n_classes > 2:
        delta_f1  = diff_results.get("f1",  -float("inf"))
        delta_acc = diff_results.get("acc", -float("inf"))
        return delta_f1 > 0 and delta_acc >= -epsilon
    else:
        delta_auc = diff_results.get("auc", -float("inf"))
        delta_acc = diff_results.get("acc", -float("inf"))
        return delta_auc > 0 and delta_acc >= -epsilon

def caafe_like_gate(diff_results, n_classes=2):
    if n_classes > 2:
        return diff_results.get("f1", -float("inf")) + diff_results.get("acc", -float("inf")) > 0
    return diff_results.get("auc", -float("inf")) + diff_results.get("acc", -float("inf")) > 0

def strict_positive_gate(diff_results, n_classes=2):
    if n_classes > 2:
        return diff_results.get("f1", -float("inf")) > 0 and diff_results.get("acc", -float("inf")) > 0
    return diff_results.get("auc", -float("inf")) > 0 and diff_results.get("acc", -float("inf")) > 0

def dynamic_control_gate(diff_results, base_results=None, n_classes=2):
    """
    动态特征接收门控 (严格使用小数制，例如 0.0005 代表 0.05%)
    """
    # 统一使用绝对小数！避免百分制造成的缩放 BUG
    MIN_DELTA = 0.0005  

    # 1. 动态计算阈值 (基于 Baseline 的不稳定性)
    if base_results is not None:
        noise_auc = base_results.get("auc_std", 0)
        noise_f1  = base_results.get("f1_std", 0)
        # 用 0.5 倍标准差作为动态阈值，下限为 MIN_DELTA
        threshold_auc = max(MIN_DELTA, noise_auc * 0.5)
        threshold_f1  = max(MIN_DELTA, noise_f1  * 0.5)
    else:
        threshold_auc = MIN_DELTA
        threshold_f1  = MIN_DELTA

    delta_auc = diff_results.get("auc", -float("inf"))
    delta_f1  = diff_results.get("f1", -float("inf"))

    # 2. 定义底线：“不伤害”原则 (容忍微小的震荡，但拒绝断崖下跌)
    auc_ok = delta_auc > -MIN_DELTA
    f1_ok  = delta_f1  > -MIN_DELTA

    # 3. 门控判定
    if n_classes > 2:
        # 多分类：F1 必须显著提升，且 AUC 绝对不能断崖式崩塌
        return (delta_f1 > threshold_f1) and auc_ok
    else:
        # 二分类：至少一个显著提升，且两者都不能断崖式崩塌
        any_improve = (delta_auc > threshold_auc) or (delta_f1 > threshold_f1)
        return any_improve and auc_ok and f1_ok