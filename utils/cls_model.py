import contextlib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from utils.data import format_x
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as tmp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ---------------------------------------------------------------------------
# MLP 模型定义
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 基础训练 / 推理
# 关键修复：把 loss.item() 从 batch 循环移出，每 epoch 只同步一次
# 原来每个 batch 都调用 .item()，CUDA stream 每 batch 被强制同步一次，
# 多 stream / 多进程并发完全没用。
# ---------------------------------------------------------------------------

def train_mlp(x_tr, y_tr, n_classes, hidden_dims=(256, 128), epochs=100,
              batch_size=256, lr=1e-3, dropout=0.3, device='cpu', random_state=42):
    torch.manual_seed(random_state)

    model = MLP(x_tr.shape[1], hidden_dims, n_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    best_loss = float('inf')
    patience = 5
    trigger = 0

    for epoch in range(epochs):
        # ✅ 用 GPU tensor 累加，不触发 CPU-GPU sync
        epoch_loss = torch.zeros(1, device=device)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()          # ← 不调用 .item()

        # ✅ 每个 epoch 只同步一次（而不是每个 batch 一次）
        epoch_loss_val = epoch_loss.item()
        if epoch_loss_val < best_loss * 0.999:
            best_loss = epoch_loss_val
            trigger = 0
        else:
            trigger += 1
        if trigger >= patience:
            break

    return model


@torch.no_grad()
def predict_mlp(model, x, batch_size=512, device='cpu'):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size
    )
    probs_list = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs_list, axis=0)


# ---------------------------------------------------------------------------
# CV fold workers
# 必须是模块级函数（spawn 要求可 pickle）
# 每个 worker 是独立进程，有自己的 CUDA context，GPU driver 做 time-multiplex
# ---------------------------------------------------------------------------

def _mlp_fold_worker(args):
    """在独立子进程中训练单折 MLP，拥有独立 CUDA context。"""
    (fold_idx, tr_idx, va_idx,
     x_combined, y_combined, n_classes,
     hidden_dims, epochs, batch_size, lr, dropout, device, random_state) = args

    clf_cv = train_mlp(
        x_combined[tr_idx], y_combined[tr_idx], n_classes,
        hidden_dims, epochs, batch_size, lr, dropout,
        device, random_state=random_state + fold_idx
    )
    probs    = predict_mlp(clf_cv, x_combined[va_idx], batch_size, device)
    preds    = probs.argmax(axis=1)
    fold_acc = accuracy_score(y_combined[va_idx], preds)
    return fold_idx, va_idx, probs, preds, fold_acc


def _xgb_fold_worker(args):
    """在独立子进程中训练单折 XGBoost GPU，拥有独立 CUDA context。"""
    (fold_idx, tr_idx, va_idx,
     x_combined, y_combined, obj, metric, random_state) = args

    x_tr = format_x(x_combined.iloc[tr_idx])
    x_va = format_x(x_combined.iloc[va_idx])
    y_tr = y_combined.iloc[tr_idx]
    y_va = y_combined.iloc[va_idx]

    clf_cv = xgb.XGBClassifier(
        objective=obj, eval_metric=metric,
        device="cuda", tree_method='gpu_hist',
        random_state=random_state,
    )
    clf_cv.fit(x_tr, y_tr, verbose=False)

    probs    = clf_cv.predict_proba(x_va)
    preds    = clf_cv.predict(x_va)
    fold_acc = accuracy_score(y_va, preds)
    return fold_idx, va_idx, probs, preds, fold_acc


# ---------------------------------------------------------------------------
# 公共 CV 调度（串行 / 多进程）
# ---------------------------------------------------------------------------

def _run_cv_folds(worker_fn, fold_args_list, n_total, n_classes, n_splits,
                  parallel_cv):
    oof_probs = np.zeros((n_total, n_classes), dtype=np.float32)
    oof_preds = np.zeros(n_total, dtype=int)
    fold_accs = [None] * n_splits

    if parallel_cv:
        # spawn：每个 fold 是独立进程，各自初始化 CUDA context
        # GPU driver 自动 time-multiplex，显存会同时被多个进程占用
        ctx = tmp.get_context('spawn')
        # with ctx.Pool(processes=n_splits) as pool:
        #     for fold_idx, va_idx, probs, preds, fold_acc in \
        #             pool.map(worker_fn, fold_args_list):
        with ThreadPoolExecutor(max_workers=n_splits) as executor:
            for fold_idx, va_idx, probs, preds, fold_acc in executor.map(worker_fn, fold_args_list):
                oof_probs[va_idx] = probs
                oof_preds[va_idx] = preds
                fold_accs[fold_idx] = fold_acc
    else:
        for args in fold_args_list:
            fold_idx, va_idx, probs, preds, fold_acc = worker_fn(args)
            oof_probs[va_idx] = probs
            oof_preds[va_idx] = preds
            fold_accs[fold_idx] = fold_acc

    return oof_probs, oof_preds, fold_accs


# ---------------------------------------------------------------------------
# eval_with_mlp
# ---------------------------------------------------------------------------

def eval_with_mlp(x_train, y_train, x_test, y_test,
                  n_splits=3, random_state=42,
                  hidden_dims=(256, 128), epochs=100,
                  batch_size=256, lr=1e-3, dropout=0.3,
                  device=None, parallel_cv=False):
    """
    parallel_cv=True：spawn 多进程，每折独占 CUDA context，
                      GPU driver 做 time-multiplex，显存会同时上涨。
    parallel_cv=False：串行（默认）。
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    x_train_arr = format_x(x_train).values.astype(np.float32)
    x_test_arr  = format_x(x_test).values.astype(np.float32)
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_arr  = y_test.values  if hasattr(y_test,  'values') else np.array(y_test)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_arr)
    x_test_scaled  = scaler.transform(x_test_arr)

    n_classes = len(np.unique(y_train_arr))

    clf_global = train_mlp(x_train_scaled, y_train_arr, n_classes,
                           hidden_dims, epochs, batch_size, lr, dropout,
                           device, random_state)
    clf_global.scaler = scaler

    x_combined = np.concatenate([x_train_scaled, x_test_scaled], axis=0)
    y_combined = np.concatenate([y_train_arr, y_test_arr], axis=0)
    n_total    = len(x_combined)

    kf          = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_args   = [
        (fold_idx, tr_idx, va_idx,
         x_combined, y_combined, n_classes,
         hidden_dims, epochs, batch_size, lr, dropout, device, random_state)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(x_combined))
    ]

    oof_probs, oof_preds, fold_accs = _run_cv_folds(
        _mlp_fold_worker, fold_args, n_total, n_classes, n_splits, parallel_cv
    )

    acc = accuracy_score(y_combined, oof_preds)
    f1  = f1_score(y_combined, oof_preds, average='weighted')
    try:
        auc = (roc_auc_score(y_combined, oof_probs[:, 1])
               if n_classes == 2
               else roc_auc_score(y_combined, oof_probs, multi_class='ovr'))
    except Exception:
        auc = -1

    return clf_global, {
        'acc':     acc              * 100,
        'auc':     auc              * 100,
        'f1':      f1               * 100,
        'acc_std': np.std(fold_accs)* 100,
    }


# ---------------------------------------------------------------------------
# eval_with_mlp_final（不涉及并行，保持不变）
# ---------------------------------------------------------------------------

def eval_with_mlp_final(x_train, y_train, x_test, y_test,
                        hidden_dims=(256, 128), epochs=100,
                        batch_size=256, lr=1e-3, dropout=0.3,
                        device=None, random_state=42):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    x_train_arr = format_x(x_train).values.astype(np.float32)
    x_test_arr  = format_x(x_test).values.astype(np.float32)
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_arr  = y_test.values  if hasattr(y_test,  'values') else np.array(y_test)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_arr)
    x_test_scaled  = scaler.transform(x_test_arr)
    n_classes = len(np.unique(y_train_arr))

    clf = train_mlp(x_train_scaled, y_train_arr, n_classes,
                    hidden_dims, epochs, batch_size, lr, dropout, device, random_state)
    clf.scaler = scaler

    probs  = predict_mlp(clf, x_test_scaled, batch_size, device)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test_arr, y_pred)
    f1  = f1_score(y_test_arr, y_pred, average='weighted')
    try:
        auc = (roc_auc_score(y_test_arr, probs[:, 1])
               if n_classes == 2
               else roc_auc_score(y_test_arr, probs, multi_class='ovr'))
    except Exception:
        auc = -1

    return clf, {'acc': acc * 100, 'auc': auc * 100, 'f1': f1 * 100}


# ---------------------------------------------------------------------------
# eval_with_xgb
# ---------------------------------------------------------------------------

def eval_with_xgb(x_train, y_train, x_test, y_test,
                  n_splits=3, random_state=42,
                  parallel_cv=False):
    """
    parallel_cv=True：spawn 多进程，每个 XGBoost GPU 实例独立 CUDA context，
                      GPU driver 做 time-multiplex。
    """
    x_train = format_x(x_train)
    x_test  = format_x(x_test)

    n_classes = len(np.unique(y_train))
    obj    = "binary:logistic" if n_classes == 2 else "multi:softprob"
    metric = 'logloss'         if n_classes == 2 else 'mlogloss'

    clf_global = xgb.XGBClassifier(
        objective=obj, eval_metric=metric,
        device="cuda", tree_method='gpu_hist', random_state=random_state
    )
    clf_global.fit(format_x(x_train), y_train, verbose=False)

    x_combined = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    n_total    = len(x_combined)

    kf        = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_args = [
        (fold_idx, tr_idx, va_idx,
         x_combined, y_combined, obj, metric, random_state)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(x_combined))
    ]

    oof_probs, oof_preds, fold_accs = _run_cv_folds(
        _xgb_fold_worker, fold_args, n_total, n_classes, n_splits, parallel_cv
    )

    y_true = y_combined.values
    acc = accuracy_score(y_true, oof_preds)
    f1  = f1_score(y_true, oof_preds, average="weighted")
    try:
        auc = (roc_auc_score(y_true, oof_probs[:, 1])
               if n_classes == 2
               else roc_auc_score(y_true, oof_probs, multi_class='ovr'))
    except Exception:
        auc = -1

    return clf_global, {
        'acc':     acc              * 100,
        'auc':     auc              * 100,
        'f1':      f1               * 100,
        'acc_std': np.std(fold_accs)* 100,
    }


def eval_with_xgb_final(x_train, y_train, x_test, y_test, random_state=42):
    x_train = format_x(x_train)
    x_test  = format_x(x_test)

    n_classes = len(np.unique(y_train))
    obj    = "binary:logistic" if n_classes == 2 else "multi:softprob"
    metric = 'logloss'         if n_classes == 2 else 'mlogloss'

    clf = xgb.XGBClassifier(
        objective=obj, eval_metric=metric,
        device="cuda", tree_method='gpu_hist', random_state=random_state
    )
    clf.fit(x_train, y_train, verbose=False)

    y_pred      = clf.predict(x_test)
    y_pred_prob = clf.predict_proba(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = (roc_auc_score(y_test, y_pred_prob[:, 1])
               if n_classes == 2
               else roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
    except Exception:
        auc = -1

    return clf, {'acc': acc * 100, 'auc': auc * 100, 'f1': f1 * 100}


# ---------------------------------------------------------------------------
# 统一对外接口
# ---------------------------------------------------------------------------

def eval_with_model(x_train, y_train, x_test, y_test,
                    task_model="xgboost", random_state=42,
                    parallel_cv=False, **kwargs):
    name = task_model.lower()
    if "xgb" in name:
        return eval_with_xgb(x_train, y_train, x_test, y_test,
                             random_state=random_state, parallel_cv=parallel_cv)
    elif "mlp" in name:
        kwargs.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return eval_with_mlp(x_train, y_train, x_test, y_test,
                             random_state=random_state, parallel_cv=parallel_cv, **kwargs)
    else:
        raise ValueError(f"不支持的 task_model: '{task_model}'")


def eval_with_model_final(x_train, y_train, x_test, y_test,
                          task_model="xgboost", random_state=42, **kwargs):
    name = task_model.lower()
    if "xgb" in name:
        return eval_with_xgb_final(x_train, y_train, x_test, y_test,
                                   random_state=random_state)
    elif "mlp" in name:
        kwargs.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return eval_with_mlp_final(x_train, y_train, x_test, y_test,
                                   random_state=random_state, **kwargs)
    else:
        raise ValueError(f"不支持的 task_model: '{task_model}'")