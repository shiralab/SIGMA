import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_datasets(dataset_path, masked=True):
    if not masked:
        train_file = 'raw_train.csv'
        val_file = 'raw_val.csv'
        test_file = 'raw_test.csv'
    else:
        train_file = 'train.csv'
        val_file = 'validation.csv'
        test_file = 'test.csv'
    
    train_ds = pd.read_csv(os.path.join(dataset_path, train_file))
    val_ds = pd.read_csv(os.path.join(dataset_path, val_file))
    test_ds = pd.read_csv(os.path.join(dataset_path, test_file))

    # 当加载原始数据时，对最后一列（Target）进行编码处理
    if not masked:
        le = LabelEncoder()
        
        # 获取最后一列的列名
        train_target_col = train_ds.columns[-1]
        val_target_col = val_ds.columns[-1]
        test_target_col = test_ds.columns[-1]
        
        # 仅在训练集上进行 fit_transform
        train_ds[train_target_col] = le.fit_transform(train_ds[train_target_col])
        
        # 在验证集和测试集上进行 transform
        val_ds[val_target_col] = le.transform(val_ds[val_target_col])
        test_ds[test_target_col] = le.transform(test_ds[test_target_col])

    return train_ds, val_ds, test_ds

def load_ori_datasets(dataset_path):
    train_ds = pd.read_csv(os.path.join(dataset_path, 'raw_train.csv'))
    val_ds = pd.read_csv(os.path.join(dataset_path, 'raw_val.csv'))
    test_ds = pd.read_csv(os.path.join(dataset_path, 'raw_test.csv'))

    return train_ds, val_ds, test_ds

def get_X_y(ds):
    X = ds.iloc[:, :-1]
    y = ds.iloc[:, -1]

    return X, y

def format_y(y):
    """
    对目标变量 y 进行工业级清洗与编码。
    
    Returns:
        y_encoded (np.ndarray): 展平且编码后的 1D numpy 数组
        task_type (str): 'binary', 'multiclass', 或 'regression'
        encoder (LabelEncoder or None): 如果进行了标签编码，返回拟合好的 encoder
    """
    # 1. 统一容器与维度 (处理 DataFrame, Series, List, 2D Array)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values
    elif isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y)

    if y.ndim > 1:
        y = y.ravel() # 强制展平为 (N,)

    # 2. 致命错误拦截：检查缺失值
    if pd.isnull(y).any():
        raise ValueError("[Error] 目标变量 `y` 中包含缺失值(NaN)。请在格式化前清洗数据！")

    # 3. 启发式判断任务类型
    encoder = None
    is_numeric = np.issubdtype(y.dtype, np.number)
    
    if not is_numeric or y.dtype == bool:
        # 非数字或布尔值 -> 绝对是分类任务
        task_type = "classification"
    else:
        # 数字类型：需要区分是连续回归，还是非连续的分类/完美编码的分类
        unique_vals = np.unique(y)
        is_integer = np.all(np.mod(unique_vals, 1) == 0)
        
        # 判断规则：如果是整数，且唯一值数量 < 样本总数的 5% 且类别数 < 100，认定为分类
        # (避免把销量预测这种整数回归任务误判为分类)
        if is_integer and (len(unique_vals) < 0.05 * len(y) and len(unique_vals) < 100):
            task_type = "classification"
        else:
            task_type = "regression"

    # 4. 执行编码与转换
    if task_type == "classification":
        unique_vals = np.unique(y)
        
        # 检查是否已经是完美的 0 到 n_classes-1 编码 (免检通道，省算力)
        is_perfect_encoded = (
            np.issubdtype(y.dtype, np.integer) and 
            len(unique_vals) > 0 and 
            unique_vals.min() == 0 and 
            unique_vals.max() == len(unique_vals) - 1
        )
        
        if not is_perfect_encoded:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        else:
            y = y.astype(int)
            
        # 进一步细分是二分类还是多分类
        n_classes = len(np.unique(y))
        task_type = "binary" if n_classes == 2 else "multiclass"
            
    elif task_type == "regression":
        # 回归任务统一转为 float
        y = y.astype(float)

    return y, task_type, encoder

def format_x(x):
    """
    check format for proper use of xgboost.
    including datatype (all should be numeric)
    values: inf, Nan
    
    return processed_x
    """
    # 1. 确保输入是 DataFrame 格式，方便处理列名和类型
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    
    # 拷贝一份数据，避免直接修改原始输入（SettingWithCopyWarning）
    processed_x = x.copy()
    
    # 2. 处理非数值类型 (Object/Category)
    # 尝试将能转换的列转为数值，不能转的会变成 NaN
    for col in processed_x.columns:
        if not np.issubdtype(processed_x[col].dtype, np.number):
            # errors='coerce' 会将无法转换的字符串等转为 NaN
            processed_x[col] = pd.to_numeric(processed_x[col], errors='coerce')
    
    # 3. 处理极值 (Infinity)
    # XGBoost 对 inf 非常敏感，通常会导致计算增益时出现错误
    # 将正负无穷替换为 NaN，让 XGBoost 利用其内置的缺失值分裂逻辑来处理
    processed_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 4. 最终类型检查
    # 确保所有数据都是 float32 或 float64，这是 XGBoost 最喜欢的格式
    processed_x = processed_x.astype(np.float32)
    
    return processed_x

def mask_feature_names(ds):
    n_features = ds.shape[1]

    ds.columns = [f'f{i}' for i in range(1, n_features+1)]

    return ds