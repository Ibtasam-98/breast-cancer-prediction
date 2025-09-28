import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# Set global font sizes for enhanced readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

os.makedirs('output', exist_ok=True)

# Define enhanced blue color palette
BLUE_PALETTE = sns.light_palette("#1f77b4", as_cmap=True)
BLUE_COLORS = sns.light_palette("#1f77b4", n_colors=15)
DARK_BLUE = "#1f77b4"
LIGHT_BLUE = "#aec7e8"


def plot_comprehensive_learning_curves(models, X_train, y_train):
    """Plot all learning curves in one comprehensive visualization with different colors and line styles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    plt.suptitle('Comprehensive Learning Curves - All Models', fontsize=22, fontweight='bold', y=0.98)

    # Define a color palette with distinct colors for each model
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]

    # Define line styles for training vs validation
    train_style = '-'
    val_style = '--'

    # Define markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # Store all data for the second plot
    final_train_scores = []
    final_val_scores = []
    model_names = []

    for i, (name, model) in enumerate(models.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        try:
            # Calculate learning curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy', random_state=42)

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Store final scores for comparison plot
            final_train_scores.append(train_mean[-1])
            final_val_scores.append(val_mean[-1])
            model_names.append(name)

            # Plot 1: Learning curves with confidence intervals
            # Training scores
            ax1.plot(train_sizes, train_mean, train_style,
                     color=color, marker=marker, markersize=6,
                     linewidth=2.5, label=f'{name} (Train)')
            ax1.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.2, color=color)

            # Validation scores
            ax1.plot(train_sizes, val_mean, val_style,
                     color=color, marker=marker, markersize=6,
                     linewidth=2.5, label=f'{name} (Val)')
            ax1.fill_between(train_sizes, val_mean - val_std,
                             val_mean + val_std, alpha=0.2, color=color)

        except Exception as e:
            print(f"⚠️ Error generating learning curve for {name}: {str(e)}")
            continue

    # Customize first subplot
    ax1.set_xlabel('Training Examples', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy Score', fontsize=16, fontweight='bold')
    ax1.set_title('Learning Curves - Training vs Validation Performance',
                  fontsize=18, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    # Set y-axis limits for better visualization
    ax1.set_ylim(0.75, 1.05)

    # Plot 2: Final performance comparison
    if final_train_scores and final_val_scores:
        x_pos = np.arange(len(model_names))
        width = 0.35

        bars1 = ax2.bar(x_pos - width / 2, final_train_scores, width,
                        label='Final Training Score', color=DARK_BLUE, alpha=0.8)
        bars2 = ax2.bar(x_pos + width / 2, final_val_scores, width,
                        label='Final Validation Score', color=LIGHT_BLUE, alpha=0.8)

        ax2.set_xlabel('Models', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Final Accuracy Score', fontsize=16, fontweight='bold')
        ax2.set_title('Final Performance Comparison',
                      fontsize=18, fontweight='bold', pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1%}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add gap analysis
        for i, (train_score, val_score) in enumerate(zip(final_train_scores, final_val_scores)):
            gap = train_score - val_score
            if gap > 0.02:  # Only show significant overfitting
                ax2.text(i, max(train_score, val_score) + 0.02, f'Δ={gap:.1%}',
                         ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/comprehensive_learning_curves.png',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved comprehensive learning curves plot")
    return fig


def plot_comprehensive_performance(results):
    """Create a comprehensive performance visualization"""
    df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=20, fontweight='bold', y=0.98)

    # Plot 1: Accuracy Comparison
    x = np.arange(len(df))
    width = 0.35
    y_min = 0.80  # Start from 80% for better visualization
    y_max = min(1.0, max(df['train_accuracy'].max(), df['test_accuracy'].max()) + 0.05)

    ax1.set_ylim(y_min, y_max)
    train_bars = ax1.bar(x - width / 2, df['train_accuracy'], width,
                         label='Training Accuracy', color=DARK_BLUE, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
    test_bars = ax1.bar(x + width / 2, df['test_accuracy'], width,
                        label='Testing Accuracy', color=LIGHT_BLUE, alpha=0.8,
                        edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax1.set_title('Training vs Testing Accuracy', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=14)
    ax1.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars in [train_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            if height >= y_min:
                ax1.annotate(f'{height:.1%}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot 2: Training Time
    ax2.bar(range(len(df)), df['training_time'], color=DARK_BLUE, alpha=0.7)
    ax2.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=16, fontweight='bold')
    ax2.set_title('Model Training Time', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Additional Metrics
    metrics_data = {
        'Sensitivity': [r['sensitivity'] for r in results],
        'Specificity': [r['specificity'] for r in results],
        'Precision': [r['precision'] for r in results]
    }
    metrics_df = pd.DataFrame(metrics_data, index=df['model'])

    x_metrics = np.arange(len(metrics_df))
    width_metrics = 0.25

    ax3.bar(x_metrics - width_metrics, metrics_df['Sensitivity'], width_metrics,
            label='Sensitivity', color='#2ca02c', alpha=0.7)
    ax3.bar(x_metrics, metrics_df['Specificity'], width_metrics,
            label='Specificity', color='#ff7f0e', alpha=0.7)
    ax3.bar(x_metrics + width_metrics, metrics_df['Precision'], width_metrics,
            label='Precision', color='#d62728', alpha=0.7)

    ax3.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax3.set_title('Additional Performance Metrics', fontsize=18, fontweight='bold', pad=20)
    ax3.set_xticks(x_metrics)
    ax3.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=14)
    ax3.legend(fontsize=14)
    ax3.set_ylim(0.7, 1.0)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Rank by Test Accuracy
    ranks = np.arange(1, len(df) + 1)
    ax4.barh(ranks, df['test_accuracy'], color=DARK_BLUE, alpha=0.7)
    ax4.set_yticks(ranks)
    ax4.set_yticklabels([f"{i}. {model}" for i, model in enumerate(df['model'], 1)], fontsize=14)
    ax4.set_xlabel('Test Accuracy', fontsize=16, fontweight='bold')
    ax4.set_title('Model Ranking by Test Accuracy', fontsize=18, fontweight='bold', pad=20)
    ax4.set_xlim(0.8, 1.0)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/comprehensive_performance.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved comprehensive performance plot")
    return fig


def plot_confusion_matrices_comprehensive(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(models)
    cols = 4
    rows = (n_models + cols - 1) // cols

    fig = plt.figure(figsize=(20, 5 * rows))
    plt.suptitle('Confusion Matrices - All Models', fontsize=20, fontweight='bold', y=0.98)

    for i, (name, model) in enumerate(models.items(), 1):
        ax = plt.subplot(rows, cols, i)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=BLUE_PALETTE, vmin=0, vmax=1)

        # Add text annotations
        for i_arr in range(cm_norm.shape[0]):
            for j_arr in range(cm_norm.shape[1]):
                ax.text(j_arr, i_arr, f"{cm[i_arr, j_arr]}\n({cm_norm[i_arr, j_arr]:.2f})",
                        ha="center", va="center", color="black" if cm_norm[i_arr, j_arr] < 0.7 else "white",
                        fontsize=12, fontweight='bold')

        ax.set_title(f'{name}', fontsize=16, fontweight='bold', pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign', 'Malignant'], fontsize=12)
        ax.set_yticklabels(['Benign', 'Malignant'], fontsize=12)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)

    plt.tight_layout()
    plt.savefig('output/confusion_matrices_comprehensive.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved comprehensive confusion matrices plot")
    return fig


def plot_roc_curves_comprehensive(models, X_test, y_test):
    """Plot ROC curves for all models in one comprehensive plot"""
    plt.figure(figsize=(12, 10))

    # Define a color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    y_test_bin = label_binarize(y_test, classes=[0, 1])

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test_bin, probas[:, 1])
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, color=color, linewidth=3,
                         label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"⚠️ Error generating ROC curve for {name}: {str(e)}")
                continue

    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=18, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig('output/roc_curves_comprehensive.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved comprehensive ROC curves plot")
    return plt.gcf()


def save_all_visualizations(models, results, X_train, y_train, X_test, y_test):
    """Generate and save all visualizations"""
    print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS...")

    # Generate all plots
    plot_comprehensive_performance(results)
    plot_comprehensive_learning_curves(models, X_train, y_train)
    plot_confusion_matrices_comprehensive(models, X_test, y_test)
    plot_roc_curves_comprehensive(models, X_test, y_test)

    print("\n✅ ALL VISUALIZATIONS SAVED:")
    print("   - output/comprehensive_performance.png")
    print("   - output/comprehensive_learning_curves.png")
    print("   - output/confusion_matrices_comprehensive.png")
    print("   - output/roc_curves_comprehensive.png")