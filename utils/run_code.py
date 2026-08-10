import os
import sys
import pandas as pd
import numpy as np
from io import StringIO


def save_code(code, results_dir, step, output_dir):
    save_dir = os.path.join(results_dir, 'code')
    save_dir = os.path.join(save_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_name = f'step_{step}.py'
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"[SAVED] Feature engineering code for Step {step} saved to {file_path}")


def _validate_feature(feature_name, func, input_df):
    """
    Generic feature validation function.
    Args:
        feature_name (str): 'intra' or 'cross' for logging
        func (callable): The feature engineering function to call
        input_df (pd.DataFrame): Input dataframe
    Returns:
        dict: {
            'success': bool,
            'col_name': str or None,
            'type': str or None,
            'df': pd.DataFrame or None
        }
    """
    default_result = {
        'success': False,
        'col_name': None,
        'type': None,
        'df': None
    }
    
    try:
        # ========== CALL FUNCTION ==========
        result = None
        try:
            result = func(input_df.copy())
        except Exception as e:
            print(f"[ERROR] {feature_name.upper()} function call failed: {type(e).__name__}: {str(e)}")
            # Try to extract col_name and type from partial result if available
            # This will be handled below
        
        # ========== VALIDATE RETURN STRUCTURE ==========
        if result is None:
            return default_result
        
        if not isinstance(result, tuple) or len(result) != 3:
            print(f"[ERROR] {feature_name.upper()} function must return tuple of 3 elements (df, col_name, type)")
            # Try to extract col_name if tuple has at least 2 elements
            if isinstance(result, tuple) and len(result) >= 2:
                col_name = result[1] if isinstance(result[1], str) else None
                feat_type = result[2] if len(result) >= 3 and isinstance(result[2], str) else None
                if col_name:
                    print(f"[INFO] Extracted col_name from partial result: '{col_name}'")
                return {
                    'success': False,
                    'col_name': col_name,
                    'type': feat_type,
                    'df': None
                }
            return default_result
        
        df_result, col_name, feat_type = result
        
        # ========== VALIDATE DATAFRAME ==========
        if not isinstance(df_result, pd.DataFrame):
            print(f"[ERROR] {feature_name.upper()} first return value must be DataFrame, got {type(df_result).__name__}")
            # Still try to return col_name and type even if df is invalid
            if isinstance(col_name, str) or isinstance(feat_type, str):
                return {
                    'success': False,
                    'col_name': col_name if isinstance(col_name, str) else None,
                    'type': feat_type if isinstance(feat_type, str) else None,
                    'df': None
                }
            return default_result
        
        # ========== VALIDATE COLUMN NAME ==========
        if not isinstance(col_name, str):
            print(f"[ERROR] {feature_name.upper()} column name must be string, got {type(col_name).__name__}")
            return {
                'success': False,
                'col_name': col_name if isinstance(col_name, str) else None,
                'type': feat_type if isinstance(feat_type, str) else None,
                'df': None
            }
        
        # ========== VALIDATE FEATURE TYPE ==========
        if not isinstance(feat_type, str):
            print(f"[ERROR] {feature_name.upper()} feature type must be string, got {type(feat_type).__name__}")
            return {
                'success': False,
                'col_name': col_name,
                'type': None,
                'df': None
            }
        
        # ========== VALIDATE COLUMN EXISTS ==========
        if col_name not in df_result.columns:
            print(f"[ERROR] {feature_name.upper()} column '{col_name}' not found in returned dataframe")
            return {
                'success': False,
                'col_name': col_name,
                'type': feat_type,
                'df': None
            }
        
        # ========== EXTRACT AND CHECK FEATURE VALUES ==========
        feature_values = df_result[col_name]
        
        # ========== CHECK FOR NaN/INF VALUES ==========
        has_nan = feature_values.isna().any()
        has_inf = np.isinf(feature_values).any()
        
        if has_nan:
            print(f"[WARNING] {feature_name.upper()} '{col_name}' contains NaN values")
        if has_inf:
            print(f"[WARNING] {feature_name.upper()} '{col_name}' contains infinite values")
        
        # ========== SUCCESS ==========
        print(f"[SUCCESS] {feature_name.upper()} feature created: '{col_name}' (type: {feat_type})")
        return {
            'success': not (has_inf or has_nan),
            'col_name': col_name,
            'type': feat_type,
            'df': df_result.copy()
        }
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in {feature_name} validation: {type(e).__name__}: {str(e)}")
        return default_result


def apply_code(code, input_df):
    """
    Safely run generated feature engineering code with two separate functions.
    Expects code to define two functions:
    1. apply_intra_group_engineering(ori_df) -> (df, col_name, type)
    2. apply_cross_group_engineering(ori_df) -> (df, col_name, type)
    
    Args:
        code (str): Python code string to execute
        input_df (pd.DataFrame): Input dataframe for feature engineering
    
    Returns:
        tuple: (success: bool, data: dict)
        data structure: {
            'intra': {
                'success': bool,
                'col_name': str or None,
                'type': str or None,
                'df': pd.DataFrame or None
            },
            'cross': {
                'success': bool,
                'col_name': str or None,
                'type': str or None,
                'df': pd.DataFrame or None
            }
        }
        Overall success=True if at least one feature is successful.
        Individual features succeed/fail independently.
    """
    # Default result structure for each feature
    default_feature = {
        'success': False,
        'col_name': None,
        'type': None,
        'df': None
    }
    
    default_data = {
        'intra': default_feature.copy(),
        'cross': default_feature.copy()
    }
    
    try:
        # ========== INPUT VALIDATION ==========
        if not isinstance(code, str) or not code.strip():
            print("[ERROR] Code must be a non-empty string")
            return False, default_data
        
        if not isinstance(input_df, pd.DataFrame) or input_df.empty:
            print("[ERROR] Input must be a non-empty pandas DataFrame")
            return False, default_data
        
        # ========== PREPARE EXECUTION ENVIRONMENT ==========
        exec_namespace = {
            'pd': pd,
            'np': np,
            'pandas': pd,
            'numpy': np,
            'ori_df': input_df.copy(),
        }
        
        # Suppress stdout/stderr during execution
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            # ========== EXECUTE CODE ==========
            exec(code, exec_namespace)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # ========== EXTRACT AND VALIDATE INTRA FUNCTION ==========
        intra_result = default_feature.copy()
        if 'apply_intra_group_engineering' not in exec_namespace:
            print("[ERROR] Function 'apply_intra_group_engineering' not found in code")
        elif not callable(exec_namespace['apply_intra_group_engineering']):
            print("[ERROR] 'apply_intra_group_engineering' is not callable")
        else:
            intra_func = exec_namespace['apply_intra_group_engineering']
            intra_result = _validate_feature('intra', intra_func, input_df)
        
        # ========== EXTRACT AND VALIDATE CROSS FUNCTION ==========
        cross_result = default_feature.copy()
        if 'apply_cross_group_engineering' not in exec_namespace:
            print("[ERROR] Function 'apply_cross_group_engineering' not found in code")
        elif not callable(exec_namespace['apply_cross_group_engineering']):
            print("[ERROR] 'apply_cross_group_engineering' is not callable")
        else:
            cross_func = exec_namespace['apply_cross_group_engineering']
            cross_result = _validate_feature('cross', cross_func, input_df)
        
        # ========== COMBINE RESULTS ==========
        final_data = {
            'intra': intra_result,
            'cross': cross_result
        }
        
        # Overall success if at least one feature succeeded
        overall_success = intra_result['success'] or cross_result['success']
        
        # Log summary
        num_success = sum([intra_result['success'], cross_result['success']])
        if overall_success:
            print(f"[RESULT] {num_success}/2 features created successfully")
        else:
            print(f"[RESULT] No features created successfully")
        
        return overall_success, final_data
        
    except SyntaxError as e:
        print(f"[ERROR] Syntax error in code: {str(e)}")
        return False, default_data
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {str(e)}")
        return False, default_data