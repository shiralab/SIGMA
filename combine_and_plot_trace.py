import os
import shutil
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_experiment_path(root_dir):
    """
    Parse path, extract experiment_name and round_n.
    Looks for the component matching 'run_round_*' pattern,
    then takes the directory before it as experiment_name.
    """
    path = Path(root_dir).resolve()
    parts = path.parts

    # Find the round directory (matches run_round_N or any round-like name)
    round_idx = None
    for i, part in enumerate(parts):
        if part.startswith('run_round_'):
            round_idx = i
            break

    if round_idx is None:
        raise ValueError(
            f"Cannot find 'run_round_*' component in path: {root_dir}\n"
            f"Path parts: {parts}"
        )

    if round_idx < 1:
        raise ValueError(f"No parent directory before round component in path: {root_dir}")

    experiment_name = parts[round_idx - 1]
    round_n = parts[round_idx]

    return experiment_name, round_n


def combine_traces(root_dir):
    """
    Combine all trace files from worker directories into a single combined_trace directory
    """
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"❌ Error: Directory not found {root_dir}")
        return root_path / "combined_trace"

    # Set output directory
    combined_dir = root_path / "combined_trace"

    # Pattern: worker_n/traces/dataset_name/seed_k/ablation_trace.csv
    search_pattern = "worker_*/traces/*/*/*/ablation_trace.csv"
    trace_files = list(root_path.glob(search_pattern))

    if not trace_files:
        print(f"⚠️ No trace files found matching pattern in {root_dir}!")
        return combined_dir

    print(f"🔍 Found {len(trace_files)} trace files, combining...\n")

    success_count = 0
    for file_path in trace_files:
        try:
            # Extract directory level names
            seed_name = file_path.parent.name
            dataset_name = file_path.parent.parent.name
            
            # Build destination folder path: root_dir/combined_trace/seed_k/
            dest_dir = combined_dir / seed_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Destination filename is the dataset name: e.g., cora.csv
            dest_file = dest_dir / f"{dataset_name}.csv"
            
            # Copy and rename
            shutil.copy2(file_path, dest_file)
            success_count += 1
            
            # Print relative paths for verification
            src_rel = file_path.relative_to(root_path)
            dest_rel = dest_file.relative_to(root_path)
            print(f"✅ Success: {src_rel}  ->  {dest_rel}")
            
        except Exception as e:
            print(f"❌ Error copying file {file_path}: {e}")

    print(f"\n🎉 Combination complete! Successfully processed {success_count} files.\n")
    return combined_dir


def plot_metrics(trace_dir, output_file='metrics_trend.png'):
    """
    Plot metrics from trace files
    """
    # 1. Get all CSV files
    trace_path = Path(trace_dir)
    # rglob('*.csv') recursively finds all CSV files
    csv_files = list(trace_path.rglob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {trace_dir}!")
        return

    print(f"📊 Found {len(csv_files)} CSV files, generating plots...\n")

    # 2. Calculate grid layout (rows and columns)
    n_datasets = len(csv_files)
    cols = math.ceil(math.sqrt(n_datasets))
    rows = math.ceil(n_datasets / cols)

    # 3. Create main figure (slightly wider to accommodate complex legends)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    
    # Flatten axes for easier iteration
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 4. Iterate through files and plot
    for i, file_path in enumerate(csv_files):
        ax = axes[i]
        
        # Parse Dataset and Seed names
        dataset_name = file_path.name.split('.')[0]
        seed_name = file_path.parent.name
        display_title = f"{dataset_name} ({seed_name})"
            
        try:
            # Read data
            df = pd.read_csv(file_path)
            
            # Ensure sorted by step to avoid zigzag lines
            if 'step' in df.columns:
                df = df.sort_values(by='step')
                x = df['step']
            else:
                x = range(len(df))
                
            # Define required columns
            required_cols = ['val_acc', 'val_auc', 'val_f1', 'test_acc', 'test_auc', 'test_f1']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column not found in CSV: {col}")

            # ==========================================
            # Plotting: Unified color scheme and line styles
            # Blue: Accuracy | Orange: AUC | Green: F1
            # Val set: Solid line + circles | Test set: Dashed line + crosses
            # ==========================================
            
            # Accuracy
            ax.plot(x, df['val_acc'],  color='tab:blue',   linestyle='-',  marker='o', markersize=3, label='Val Acc')
            ax.plot(x, df['test_acc'], color='tab:blue',   linestyle='--', marker='x', markersize=4, label='Test Acc', alpha=0.8)
            
            # AUC
            ax.plot(x, df['val_auc'],  color='tab:orange', linestyle='-',  marker='o', markersize=3, label='Val AUC')
            ax.plot(x, df['test_auc'], color='tab:orange', linestyle='--', marker='x', markersize=4, label='Test AUC', alpha=0.8)
            
            # F1 Score
            ax.plot(x, df['val_f1'],   color='tab:green',  linestyle='-',  marker='o', markersize=3, label='Val F1')
            ax.plot(x, df['test_f1'],  color='tab:green',  linestyle='--', marker='x', markersize=4, label='Test F1', alpha=0.8)

            # Set subplot properties
            ax.set_title(f'Dataset: {display_title}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle=':', alpha=0.7)
            
            # Legend with smaller font, two columns
            ax.legend(loc='best', fontsize='small', ncol=2)

        except Exception as e:
            ax.set_title(f'Error reading {display_title}')
            ax.text(0.5, 0.5, str(e), ha='center', va='center', color='red')
            print(f"❌ Error processing file {file_path}: {e}")

    # 5. Hide extra blank subplots
    for j in range(n_datasets, len(axes)):
        fig.delaxes(axes[j])

    # 6. Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plotting complete! Figure saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine trace files and plot metrics. Input path should be: results/main/{experiment_name}/{round_n}/..."
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True, 
        help="Root directory path (format: results/main/{experiment_name}/{round_n}/...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="", 
        help="output_dir (format: results/main/{experiment_name}/{round_n}/...)"

    )
    
    args = parser.parse_args()
    
    try:
        # Parse experiment_name and round_n from path
        experiment_name, round_n = parse_experiment_path(args.root_dir)
        print(f"📁 Experiment: {experiment_name}, Round: {round_n}\n")
        
        # Ensure output directory exists
        output_dir = Path("results/trace_plot")
        if args.output_dir:
            output_dir = output_dir / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_file = output_dir / f"{experiment_name}_{round_n}.png"
        
        # Step 1: Combine traces
        print("=" * 60)
        print("STEP 1: Combining traces...")
        print("=" * 60)
        combined_dir = combine_traces(args.root_dir)
        
        # Step 2: Plot metrics
        print("=" * 60)
        print("STEP 2: Plotting metrics...")
        print("=" * 60)
        plot_metrics(str(combined_dir), str(output_file))
        
        print("=" * 60)
        print("✅ All done!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"❌ Please provide a path in format: results/main/{{experiment_name}}/{{round_n}}/...")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
