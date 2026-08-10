import pandas as pd

"""
<CLS_MODEL>: classification model, from input
<N>-classes: number of classes, from y_train
<IMP_TYPE>: importance index, from input
<TOP_FEATURES>: top features information, processing
<USEFUL_FEATURES>: features better than noise, processing
<WEAK_FEATURES>: features weak than noise, processing
<SCORE>: evaluation metrics, from model
<PREVIOUS_REPORT>: last step report, including generated features and score
<TOP_5>: Top 5 strategies
<REJECT_FEATURES>: generated or deleted features
"""


def _build_rank_dict(importance_df):
    if importance_df is None or importance_df.empty:
        return {}

    return {
        feat: idx + 1
        for idx, feat in enumerate(importance_df['feature'].tolist())
    }


def _build_shap_dict(importance_df):
    if importance_df is None or importance_df.empty:
        return {}

    return dict(zip(importance_df['feature'], importance_df['shap_value']))


def _format_importance_value(feat, shap_dict, rank_dict, noise_threshold, info_mode, signed=False):
    shap_val = shap_dict.get(feat, 0.0)

    if info_mode == 'noise_diff':
        diff = shap_val - (noise_threshold or 0.0)
        return f"{'+' if signed and diff > 0 else ''}{diff:.4f}"

    if info_mode == 'raw_shap':
        return f"{shap_val:.4f}"

    if info_mode == 'rank_only':
        rank = rank_dict.get(feat, 'NA')
        return f"rank {rank}"

    if info_mode == 'basic':
        return None

    raise ValueError(f"Unsupported info_mode: {info_mode}")


def build_normal_features_inform(df, importance_df, target_features, noise_threshold, info_mode='noise_diff'):
    """
    used for top & useful features.
    format:
    `{feature_name}: {importance_diff}, {value_range}`
    - `importance_diff`: Difference from Noise baseline ("+" = higher, "-" = lower)
    - `value_range`: (Max, Min)
    """
    if not target_features:
        return "None"
    
    inform_lines = []
    shap_dict = _build_shap_dict(importance_df)
    rank_dict = _build_rank_dict(importance_df)
    
    for feat in target_features:
        if feat not in df.columns:
            continue

        f_max = df[feat].max()
        f_min = df[feat].min()
        range_str = f"({f_max:.4f}, {f_min:.4f})"

        metric_str = _format_importance_value(
            feat=feat,
            shap_dict=shap_dict,
            rank_dict=rank_dict,
            noise_threshold=noise_threshold,
            info_mode=info_mode,
            signed=True
        )

        if metric_str is None:
            line = f"{feat}: {range_str}"
        else:
            line = f"{feat}: {metric_str}, {range_str}"
        inform_lines.append(line)
    
    return "\n".join(inform_lines)

def get_samples(df, feature_name, samples=5, random_state=42):
    """
    random choose 5 samples from selected feature.
    since all the features have been encoded to numbers, it is convient.
    but, we shoud format the "float" type to avoid be to long.
    """
    if feature_name not in df.columns:
        return "[]"
    
    actual_samples = min(samples, len(df))
    series = df[feature_name].dropna()
    if series.empty:
        return "[]"

    actual_samples = min(actual_samples, len(series))
    sample_values = series.sample(n=actual_samples, random_state=random_state).tolist()

    formatted_samples = []
    for val in sample_values:
        if isinstance(val, (float, int)):
            formatted_samples.append(f"{val:.4g}")
        else:
            formatted_samples.append(str(val))
            
    return f"[{', '.join(formatted_samples)}]"

import pandas as pd

def build_detailed_features_inform(df, importance_df, target_features, noise_threshold, info_mode='noise_diff'):
    """
    used for weak features.
    format:
    `{feature_name}: {importance_diff}, {unique_values/total_values}, {data_type}, [{value_range}], {sample_values}`
    """

    if not target_features:
        return "None"
    
    inform_lines = []
    shap_dict = _build_shap_dict(importance_df)
    rank_dict = _build_rank_dict(importance_df)
    total_values = len(df)

    for feat in target_features:
        if feat not in df.columns:
            continue

        unique_count = df[feat].nunique()
        unique_ratio_str = f"{unique_count}/{total_values}"

        data_type = str(df[feat].dtype)
        sample_values = get_samples(df, feat)

        metric_str = _format_importance_value(
            feat=feat,
            shap_dict=shap_dict,
            rank_dict=rank_dict,
            noise_threshold=noise_threshold,
            info_mode=info_mode,
            signed=False
        )

        # 基础信息：唯一值比例，数据类型
        detail_parts = [unique_ratio_str, data_type]
        
        # 核心修改：仅对数值类型计算和添加 value_range
        if pd.api.types.is_numeric_dtype(df[feat]):
            f_max = df[feat].max()
            f_min = df[feat].min()
            range_str = f"({f_max:.4g}, {f_min:.4g})"
            detail_parts.append(range_str)
            
        # 添加采样值
        detail_parts.append(sample_values)

        # 如果有重要性得分，插入到最前面
        if metric_str is not None:
            detail_parts.insert(0, metric_str)

        line = f"{feat}: {', '.join(detail_parts)}"
        inform_lines.append(line)

    return "\n".join(inform_lines)

def build_feature_blocks(section_pairs):
    """
    Build numbered markdown sections for the prompt template.

    Args:
        section_pairs (list[tuple[str, str]]): Ordered list of (title, content)
    """
    blocks = []

    for idx, (title, content) in enumerate(section_pairs, start=1):
        block_content = content if content else "None"
        blocks.append(f"#### {idx}. {title}\n{block_content}")

    return "\n\n".join(blocks)

def build_score_inform(score_type, results):
    """
    score_type: "acc", "auc", "f1", "all"
    the format for 'all': 
    "ACC": acc , "AUC": auc, "f1": f1  
    """
    score_info = ''
    if score_type == 'all':
        # 统一内部使用双引号 " "，外部使用单引号 ' ' 即可避免冲突
        score_info = (
            f'"ACC": {results["acc"]:.4f}, '
            f'"AUC": {results["auc"]:.4f}, '
            f'"F1": {results["f1"]:.4f}'
        )
    else:
        # 增加一个 .get() 或检查，防止 score_type 不在字典里导致 KeyError
        val = results.get(score_type, 0)
        score_info = f"{score_type.upper()}: {val:.4f}"
    
    return score_info

def build_previous_report(previous_results):
    """
    input:
        previous_results["intra"] = {
                    "type": "intra",
                    "col_name": intra_feature,
                    "score_diff": "ERROR"
                }
        previous_results["cross"] = {
                    "type": "cross",
                    "col_name": cross_feature,
                    "score_diff": "ERROR"
                }
    format:
    type: {TYPE}, col_name: {COL_NAME}, status: {STATUS}
    type: {TYPE}, col_name: {COL_NAME}, status: {STATUS}
    """
    report_lines = []

    for key in ["intra", "cross"]:
        if key in previous_results:
            data = previous_results[key]
            col_name = data.get('col_name', 'Unknown')
            feat_type = data.get('type', key)
            status = data.get('status', 'unknown')
            line = f"type: {feat_type}, col_name: {col_name}, status: {status}"
            report_lines.append(line)

    return "\n".join(report_lines)

def build_top_k_report(generated_features, k=5):
    """
    input:
    generated_features = [{
                    "type": "cross",
                    "col_name": cross_feature,
                    "score_diff": "score"
                }]

    return format:
    recent 1: type {TYPE}, col_name: {COL_NAME}
    ...
    recent k: type {TYPE}, col_name: {COL_NAME}
    """
    if not generated_features:
        return "No accepted features found."

    report_lines = []
    recent_features = list(reversed(generated_features[-k:]))
    for i, feat in enumerate(recent_features, 1):
        line = f"recent {i}: type {feat['type']}, col_name: {feat['col_name']}"
        report_lines.append(line)

    return "\n".join(report_lines)

def fill_prompt_template(cls_model, n_classes, imp_type, top_features, useful_features,
                        weak_features, score, previous_report, top_5, reject_features,
                        operations_info,
                        prompt_path='./prompt_template/user_template_3.txt',
                        extra_replacements=None):
    """
    Fill the prompt template with actual values from current iteration.
    
    TODO: Read template from prompt_template/user_template.txt
    TODO: Replace all placeholders (<CLS_MODEL>, <N>, etc.) with corresponding values
    TODO: Return filled prompt string
    """
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    replacements = {
        '<CLS_MODEL>': cls_model,
        '<N>': str(n_classes),
        '<IMP_TYPE>': imp_type,
        '<TOP_FEATURES>': str(top_features),
        '<USEFUL_FEATURES>': str(useful_features),
        '<WEAK_FEATURES>': str(weak_features),
        '<SCORE>': str(score),
        '<PREVIOUS_REPORT>': str(previous_report),
        '<TOP_5>': str(top_5),
        '<REJECT_FEATURES>': str(reject_features),
        '<OPERATIONS_INFO>': str(operations_info),
    }

    if extra_replacements:
        replacements.update({
            f'<{key}>': str(value)
            for key, value in extra_replacements.items()
        })

    for placeholder, value in replacements.items():
        prompt_template = prompt_template.replace(placeholder, value)
    
    return prompt_template
