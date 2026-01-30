"""
Compare Models Script
Trains and compares GRU, CNN, and GRU_CNN models on the same dataset


This script:
1. Loads data once (no leakage)
2. Trains each model type with same hyperparameters
3. Evaluates all on the same test set
4. Generates comparison report with metrics and plots


Requires: TensorFlow installed
"""


from tc_forecast import TCDataLoader, TCForecastModel, TCTrainer, load_config
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sys.path.insert(0, str(Path(__file__).parent))


def train_and_evaluate_model(model_type: str, config, data, save_dir: Path):
    """
    Train and evaluate a single model

    Args:
        model_type: One of 'GRU', 'CNN', 'GRU_CNN'
        config: Configuration object
        data: Dictionary with train/val/test data and metadata
        save_dir: Directory to save results

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print(f"Training {model_type} Model")
    print("=" * 70)

    # Override model type
    config.model.type = model_type

    # Build model
    print(f"\n1. Building {model_type} model...")
    model = TCForecastModel(config)
    keras_model = model.build()
    print(f" Model built: {model.count_parameters():,} parameters")

    # Create trainer
    trainer = TCTrainer(config, model)
    trainer.output_path = save_dir / model_type.lower()
    trainer.output_path.mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\n2. Training {model_type}...")
    history = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )

    # Evaluate
    print(f"\n3. Evaluating {model_type} on test set...")
    results = trainer.evaluate(
        data['X_test'], data['y_test'],
        data['test_metadata'], data['data_loader']
    )

    # Save history
    history_path = trainer.output_path / f"{model_type.lower()}_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        history_dict = {k: [float(v) for v in vals]
                        for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)

    print(f"\n {model_type} training complete!")
    print(f"  - Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"  - Final test MAE: {results.get('mae', 'N/A')}")

    return {
        'model_type': model_type,
        'parameters': model.count_parameters(),
        'history': history.history,
        'results': results,
        'best_val_loss': min(history.history['val_loss']),
        'final_epoch': len(history.history['loss'])
    }


def plot_comparison(all_results, save_dir: Path):
    """
    Create comparison plots

    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plots
    """
    print("\nGenerating comparison plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison: GRU vs CNN vs GRU_CNN',
                 fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for result in all_results:
        model_type = result['model_type']
        history = result['history']
        ax.plot(history['loss'], label=f"{model_type} Train", alpha=0.7)
        ax.plot(history['val_loss'],
                label=f"{model_type} Val", linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final Test Metrics
    ax = axes[0, 1]
    model_names = [r['model_type'] for r in all_results]
    test_maes = [r['results'].get('mae', 0) for r in all_results]
    test_rmses = [r['results'].get('rmse', 0) for r in all_results]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, test_maes, width, label='MAE', alpha=0.8)
    ax.bar(x + width/2, test_rmses, width, label='RMSE', alpha=0.8)
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Error')
    ax.set_title('Test Set Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Distance Errors by Forecast Step
    ax = axes[1, 0]
    for result in all_results:
        model_type = result['model_type']
        if 'distance_errors' in result['results']:
            mean_errors = result['results']['distance_errors']['mean']
            timesteps = np.arange(1, len(mean_errors) + 1)
            ax.plot(timesteps, mean_errors, marker='o', label=model_type)
    ax.set_xlabel('Forecast Timestep (3-hour intervals)')
    ax.set_ylabel('Distance Error (km)')
    ax.set_title('Mean Distance Error by Forecast Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Model Complexity vs Performance
    ax = axes[1, 1]
    params = [r['parameters'] for r in all_results]
    best_losses = [r['best_val_loss'] for r in all_results]
    colors = ['blue', 'green', 'red']

    for i, result in enumerate(all_results):
        ax.scatter(params[i], best_losses[i],
                   s=200, alpha=0.6, c=colors[i],
                   label=result['model_type'])
        ax.annotate(result['model_type'],
                    (params[i], best_losses[i]),
                    textcoords="offset points",
                    xytext=(0, 10), ha='center')

    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Model Complexity vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plot_path = save_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" Comparison plot saved to: {plot_path}")

    plt.close()


def generate_report(all_results, save_dir: Path):
    """
    Generate markdown comparison report

    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save report
    """
    print("\nGenerating comparison report...")

    report = []
    report.append("# TC Forecast Model Comparison Report\n")
    report.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Summary\n\n")
    report.append(
        "| Model | Parameters | Best Val Loss | Test MAE | Test RMSE | Training Epochs |\n")
    report.append(
        "|-------|-----------|---------------|----------|-----------|----------------|\n")

    for result in all_results:
        model_type = result['model_type']
        params = f"{result['parameters']:,}"
        best_val = f"{result['best_val_loss']:.4f}"
        test_mae = f"{result['results'].get('mae', 0):.4f}"
        test_rmse = f"{result['results'].get('rmse', 0):.4f}"
        epochs = result['final_epoch']
        report.append(
            f"| {model_type} | {params} | {best_val} | {test_mae} | {test_rmse} | {epochs} |\n")

    report.append("\n## Distance Errors (72-hour forecast)\n\n")
    report.append("| Model | 24h (km) | 48h (km) | 72h (km) |\n")
    report.append("|-------|----------|----------|----------|\n")

    for result in all_results:
        model_type = result['model_type']
        if 'distance_errors' in result['results']:
            mean_errors = result['results']['distance_errors']['mean']
            # Timestep 8 = 24h, 16 = 48h, 24 = 72h (assuming 3-hour intervals)
            e_24h = mean_errors[7] if len(mean_errors) > 7 else 'N/A'
            e_48h = mean_errors[11] if len(mean_errors) > 11 else 'N/A'
            e_72h = mean_errors[-1]

            if isinstance(e_24h, float):
                report.append(
                    f"| {model_type} | {e_24h:.2f} | {e_48h:.2f} | {e_72h:.2f} |\n")
            else:
                report.append(
                    f"| {model_type} | {e_24h} | {e_48h} | {e_72h} |\n")

    report.append("\n## Recommendations\n\n")

    # Find best model by val loss
    best_model = min(all_results, key=lambda x: x['best_val_loss'])
    report.append(f"**Best Overall Model**: {best_model['model_type']}\n")
    report.append(
        f"- Lowest validation loss: {best_model['best_val_loss']:.4f}\n")
    report.append(f"- Test MAE: {best_model['results'].get('mae', 'N/A')}\n\n")

    # Find most efficient (params vs performance)
    report.append("**Model Characteristics**:\n\n")
    for result in all_results:
        model_type = result['model_type']
        if model_type == 'GRU':
            report.append(
                f"- **{model_type}**: Good for temporal patterns, lowest parameters\n")
        elif model_type == 'CNN':
            report.append(
                f"- **{model_type}**: Good for local patterns, fast inference\n")
        elif model_type == 'GRU_CNN':
            report.append(
                f"- **{model_type}**: Hybrid approach, captures both spatial and temporal\n")

    report.append("\n## Visualization\n\n")
    report.append("![Model Comparison](model_comparison.png)\n\n")

    # Save report
    report_path = save_dir / 'COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f" Report saved to: {report_path}")


def main():
    """Main comparison workflow"""

    print("\n" + "=" * 70)
    print("  TC FORECAST MODEL COMPARISON")
    print("  Training GRU, CNN, and GRU_CNN on same dataset")
    print("=" * 70)

    # Setup
    config_path = Path(__file__).parent / "tc_forecast" / "config.yaml"
    config = load_config(config_path)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(__file__).parent / "comparison_results" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {save_dir}")

    # Load data ONCE (critical for fair comparison)
    print("\nLoading data...")
    data_loader = TCDataLoader(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_and_prepare()

    print(
        f" Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Package data
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'test_metadata': data_loader.metadata_test,
        'data_loader': data_loader
    }

    # Train each model
    all_results = []

    for model_type in ['GRU', 'CNN', 'GRU_CNN']:
        try:
            result = train_and_evaluate_model(
                model_type, config, data, save_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n No models trained successfully!")
        return 1

    # Generate comparison
    try:
        plot_comparison(all_results, save_dir)
        generate_report(all_results, save_dir)
    except Exception as e:
        print(f"\n⚠ Warning: Could not generate comparison: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {save_dir}")
    print(f"\nModels trained: {len(all_results)}/{3}")

    if len(all_results) == 3:
        best = min(all_results, key=lambda x: x['best_val_loss'])
        print(f"\n Best Model: {best['model_type']}")
        print(f"  - Validation Loss: {best['best_val_loss']:.4f}")
        print(f"  - Parameters: {best['parameters']:,}")

    print(f"\n Check COMPARISON_REPORT.md for detailed analysis")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
