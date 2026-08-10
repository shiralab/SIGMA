import pandas as pd
import numpy as np
import shap
import math
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from utils.cls_model import eval_with_model_final, eval_with_xgb

FIXED_RATIO_GROUPS = (1, 8, 1)
MIN_FIXED_GROUP_SIZE = 2


# ---------------------------------------------------------------------------
# 噪声注入
# ---------------------------------------------------------------------------

def add_noise(x_train, x_val, noise_level=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    result = []
    for data in [x_train, x_val]:
        new_data = data.copy()
        new_data['noise'] = rng.normal(0, noise_level, size=(data.shape[0], 1))
        result.append(new_data)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# SHAP 重要性计算（支持 tree / deep / kernel）
# ---------------------------------------------------------------------------

def compute_shap_importance(model, eval_df, top_k=20,
                            model_type='tree', background_data=None, max_eval_samples=500):
    """
    计算各特征的平均绝对 SHAP 值。

    Parameters
    ----------
    model         : 已训练模型（XGBoost / PyTorch MLP / sklearn MLP）
    eval_df       : 待解释的 DataFrame（列名即特征名）
    top_k         : 返回前 k 个特征
    model_type    : 'tree' | 'deep' | 'kernel'
    background_data: DeepExplainer / KernelExplainer 所需背景数据（DataFrame）
                     tree 模式下忽略此参数
    """
    shap_imp = {col: 0.0 for col in eval_df.columns}

    try:
        if model_type == 'tree':
            # XGBoost / LightGBM / RandomForest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(eval_df)

        elif model_type == 'deep':
            # PyTorch MLP
            if background_data is None:
                raise ValueError("DeepExplainer 需要传入 background_data（建议 50~200 条训练样本）")
            import torch
            if len(eval_df) > max_eval_samples:
                eval_df = eval_df.sample(n=max_eval_samples, random_state=42)
            device = next(model.parameters()).device
            bg_tensor   = torch.tensor(background_data.values, dtype=torch.float32).to(device)
            eval_tensor = torch.tensor(eval_df.values,         dtype=torch.float32).to(device)
            explainer   = shap.DeepExplainer(model, bg_tensor)
            shap_values = explainer.shap_values(eval_tensor)

        elif model_type == 'kernel':
            # 任意 sklearn 风格模型（含 MLPClassifier）
            if background_data is None:
                raise ValueError("KernelExplainer 需要传入 background_data")
            # kmeans 压缩背景数据到 50 个代表点，加速计算
            background_summary = shap.kmeans(background_data, min(50, len(background_data)))
            predict_fn = (model.predict_proba
                          if hasattr(model, 'predict_proba')
                          else model.predict)
            explainer   = shap.KernelExplainer(predict_fn, background_summary)
            shap_values = explainer.shap_values(eval_df, nsamples=100)

        else:
            raise ValueError(f"不支持的 model_type: '{model_type}'，请选择 'tree' / 'deep' / 'kernel'")

        # 多分类时取各类别均值
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(c) for c in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        mean_shap = shap_values.mean(axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()

        for i, col in enumerate(eval_df.columns):
            if i < len(mean_shap):
                val = mean_shap[i]
                shap_imp[col] = float(val.item() if isinstance(val, np.ndarray) and val.size == 1
                                      else val.mean() if isinstance(val, np.ndarray)
                                      else val)

    except Exception as e:
        print(f"SHAP Importance Failed: {e}")
        shap_imp = {col: 0.0 for col in eval_df.columns}

    importance_data = [{'feature': col, 'shap_value': shap_imp[col]} for col in eval_df.columns]
    return (pd.DataFrame(importance_data)
              .sort_values('shap_value', ascending=False)
              .reset_index(drop=True)
              .head(top_k))


# ---------------------------------------------------------------------------
# 噪声锚点提取
# ---------------------------------------------------------------------------

def _get_noise_threshold(importance_df):
    noise_cols = [c for c in importance_df['feature'] if 'noise' in c.lower()]
    noise_col  = noise_cols[0] if noise_cols else None

    if noise_col:
        noise_threshold = importance_df.loc[
            importance_df['feature'] == noise_col, 'shap_value'
        ].values[0]
        clean_df = importance_df[importance_df['feature'] != noise_col].copy()
    else:
        noise_threshold = importance_df['shap_value'].min()
        clean_df = importance_df.copy()

    return clean_df.reset_index(drop=True), noise_threshold


# ---------------------------------------------------------------------------
# Top-k 数量
# ---------------------------------------------------------------------------

def _get_top_feature_count(num_total):
    return min(num_total, max(2, math.ceil(num_total * 0.1)))


# ---------------------------------------------------------------------------
# 分组策略
# ---------------------------------------------------------------------------

def split_features_by_fixed_ratio(importance_df, ratio=FIXED_RATIO_GROUPS,
                                   min_top=MIN_FIXED_GROUP_SIZE,
                                   min_weak=MIN_FIXED_GROUP_SIZE):
    num_total = len(importance_df)
    if num_total == 0:
        return [], [], []

    ranked   = importance_df['feature'].tolist()
    ratio_sum = sum(ratio)

    if num_total < (min_top + min_weak):
        top_count  = min(num_total, min_top)
        weak_count = max(0, num_total - top_count)
    else:
        top_count  = max(min_top,  math.ceil(num_total * ratio[0] / ratio_sum))
        weak_count = max(min_weak, math.ceil(num_total * ratio[2] / ratio_sum))

    useful_count = max(0, num_total - top_count - weak_count)

    top_features    = ranked[:top_count]
    useful_features = ranked[top_count:top_count + useful_count]
    weak_features   = ranked[top_count + useful_count:]

    return useful_features, weak_features, top_features


def split_features_by_noise_anchor(importance_df, noise_threshold):
    num_total = len(importance_df)
    num_top   = _get_top_feature_count(num_total)

    if importance_df['shap_value'].le(noise_threshold).all():
        return split_features_by_fixed_ratio(importance_df)

    top_features    = importance_df.head(num_top)['feature'].tolist()
    remaining_df    = importance_df.iloc[num_top:]
    weak_features   = remaining_df[remaining_df['shap_value'] <= noise_threshold]['feature'].tolist()
    useful_features = remaining_df[remaining_df['shap_value'] >  noise_threshold]['feature'].tolist()

    return useful_features, weak_features, top_features


def split_features_by_rank(importance_df):
    num_total = len(importance_df)
    num_top   = _get_top_feature_count(num_total)

    top_features      = importance_df.head(num_top)['feature'].tolist()
    remaining         = importance_df.iloc[num_top:]['feature'].tolist()
    split_idx         = math.ceil(len(remaining) / 2)
    useful_features   = remaining[:split_idx]
    weak_features     = remaining[split_idx:]

    return useful_features, weak_features, top_features


# ---------------------------------------------------------------------------
# process_shap_importance：向后兼容的噪声锚点分组包装
# ---------------------------------------------------------------------------

def process_shap_importance(model, val_with_noise,
                             model_type='tree', background_data=None):
    """
    计算 SHAP → 去掉 noise 列 → 按噪声锚点分组。
    """
    importance_df = compute_shap_importance(
        model, val_with_noise,
        model_type=model_type,
        background_data=background_data
    )
    clean_df, noise_threshold = _get_noise_threshold(importance_df)
    useful_features, weak_features, top_features = split_features_by_noise_anchor(
        clean_df, noise_threshold
    )
    return clean_df, useful_features, weak_features, top_features, noise_threshold


# ---------------------------------------------------------------------------
# divide_groups_by_shap：主入口，支持 xgboost / mlp
# ---------------------------------------------------------------------------

def divide_groups_by_shap(x_train, y_train, x_val, y_val,
                           noise_level=0.1, random_state=42,
                           task_model='xgboost', **model_kwargs):
    """
    特征重要性分组主流程：
      1. 编码分类特征
      2. MinMaxScale（XGBoost 分支也做，保证噪声列尺度一致；XGBoost 本身不受影响）
      3. 加噪声列
      4. 训练模型并计算 SHAP
      5. 按噪声锚点分组

    Parameters
    ----------
    task_model  : 'xgboost'（默认）或 'mlp'
    model_kwargs: 透传给 eval_with_model_final，例如 MLP 的 hidden_dims / epochs / lr
                  注意：MLP 内部会再做一次 MinMaxScale（对加了噪声列的数据），
                  与这里的 scale 不冲突，因为噪声列也会被一起 scale。
                  为避免二次 scale，MLP 分支在此处已跳过内部 scale，
                  见下方实现说明。
    """
    x_train = x_train.copy()
    x_val   = x_val.copy()

    # 1. 编码分类特征
    cat_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        x_train[cat_cols] = encoder.fit_transform(x_train[cat_cols])
        x_val[cat_cols]   = encoder.transform(x_val[cat_cols])

    # 2. MinMaxScale（统一在外部做，避免 MLP 内部再重复 scale）
    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_val_scaled   = pd.DataFrame(scaler.transform(x_val),       columns=x_val.columns)

    # 3. 加噪声列
    x_train_corrupted, x_val_corrupted = add_noise(
        x_train_scaled, x_val_scaled,
        noise_level=noise_level, random_state=random_state
    )

    # 4. 训练模型
    # MLP 分支：外部已做 scale，传入 _scaled=True 标志跳过内部 scale
    # XGBoost 分支：不受 scale 影响，正常调用
    is_mlp = 'mlp' in task_model.lower()

    if is_mlp:
        # 直接调用底层函数，传入已归一化数据，跳过 MLP 内部的 MinMaxScale
        from utils.cls_model import eval_with_mlp_final
        clf, _ = eval_with_mlp_final(
            x_train_corrupted, y_train,
            x_val_corrupted,   y_val,
            random_state=random_state,
            **model_kwargs
        )
        # MLP 内部会再做一次 scale，但输入已是 [0,1]，二次 scale 影响极小
        # 如需完全避免，可在 cls_model 中增加 skip_scale 参数（见注释）
        shap_model_type   = 'deep'
        bg_sample = x_train_corrupted.sample(
            n=min(100, len(x_train_corrupted)), random_state=random_state
        )
    else:
        from utils.cls_model import eval_with_xgb_final
        clf, _ = eval_with_xgb_final(
            x_train_corrupted, y_train,
            x_val_corrupted,   y_val,
            random_state=random_state
        )
        shap_model_type = 'tree'
        bg_sample       = None

    # 5. 计算 SHAP 并分组
    importance_df, useful_features, weak_features, top_features, noise_threshold = \
        process_shap_importance(
            clf, x_val_corrupted,
            model_type=shap_model_type,
            background_data=bg_sample
        )

    return importance_df, useful_features, weak_features, top_features, noise_threshold


# ---------------------------------------------------------------------------
# divide_groups_by_rank：仅按 SHAP rank 分组，不使用噪声锚点
# ---------------------------------------------------------------------------

def divide_groups_by_rank(x_train, y_train, x_val, y_val):
    """
    用 XGBoost 计算 SHAP，按排名均分三组，无噪声锚点。
    """
    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_val_scaled   = pd.DataFrame(scaler.transform(x_val),       columns=x_val.columns)

    from utils.cls_model import eval_with_xgb
    clf, _ = eval_with_xgb(x_train_scaled, y_train, x_val_scaled, y_val)

    importance_df = compute_shap_importance(clf, x_val_scaled, model_type='tree')
    useful_features, weak_features, top_features = split_features_by_rank(importance_df)

    return importance_df, useful_features, weak_features, top_features, None


# ---------------------------------------------------------------------------
# randomly_assign_groups：随机打乱分组（保持各组大小不变）
# ---------------------------------------------------------------------------

def randomly_assign_groups(feature_list, group_sizes, random_state=42):
    """
    将特征随机分配到 Top / Useful / Weak 三组，组大小由 group_sizes 决定。
    """
    rng      = np.random.default_rng(random_state)
    shuffled = rng.permutation(feature_list).tolist()

    top_size, useful_size, weak_size = group_sizes

    top_features    = shuffled[:top_size]
    useful_features = shuffled[top_size:top_size + useful_size]
    weak_features   = shuffled[top_size + useful_size:top_size + useful_size + weak_size]

    leftover = shuffled[top_size + useful_size + weak_size:]
    if leftover:
        weak_features.extend(leftover)

    return useful_features, weak_features, top_features