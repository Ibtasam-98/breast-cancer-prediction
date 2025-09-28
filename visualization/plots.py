import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
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

# New blue shades palette for 7 models
BLUE_SHADES = [
    '#1f77b4',  # Dark Blue
    '#4292c6',  # Blue
    '#6baed6',  # Medium Blue
    '#9ecae1',  # Light Blue
    '#c6dbef',  # Very Light Blue
    '#08306b',  # Navy Blue
    '#2171b5'  # Royal Blue
]


def plot_comprehensive_learning_curves_single(models, X_train, y_train):
    """Single plot learning curves for all models with blue shades"""
    plt.figure(figsize=(14, 10))

    for i, (name, model) in enumerate(models.items()):
        color = BLUE_SHADES[i % len(BLUE_SHADES)]

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

            # Plot learning curves
            plt.plot(train_sizes, train_mean, '-', color=color, linewidth=2.5,
                     label=f'{name} (Train)', alpha=0.8)
            plt.plot(train_sizes, val_mean, '--', color=color, linewidth=2.5,
                     label=f'{name} (Val)', alpha=0.8)

            # Add confidence intervals
            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.1, color=color)
            plt.fill_between(train_sizes, val_mean - val_std,
                             val_mean + val_std, alpha=0.1, color=color)

        except Exception as e:
            print(f"⚠️ Error generating learning curve for {name}: {str(e)}")
            continue

    plt.xlabel('Training Examples', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=16, fontweight='bold')
    plt.title('Learning Curves - All Models (Training vs Validation)',
              fontsize=18, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(0.75, 1.05)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    plt.tight_layout()
    plt.savefig('output/learning_curves_single.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved single learning curves plot")


def plot_feature_importance_single(models, feature_names, X_test, y_test):
    """Single plot showing feature importance comparison for all models"""
    plt.figure(figsize=(16, 10))

    # Get top 10 features across all models
    all_importances = []

    for model_name, model in models.items():
        if model_name in ['SGD Classifier', 'XGBoost', 'AdaBoost', 'Random Forest Optimized']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = permutation_importance(
                    model, X_test, y_test, n_repeats=10, random_state=42
                )

            # Store feature importances
            for i, importance in enumerate(result.importances_mean):
                all_importances.append({
                    'feature': feature_names[i],
                    'importance': importance,
                    'model': model_name
                })

    # Create DataFrame and get top 10 features by average importance
    importance_df = pd.DataFrame(all_importances)
    top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10).index

    # Filter data for top features
    filtered_df = importance_df[importance_df['feature'].isin(top_features)]

    # Create grouped bar plot
    pivot_df = filtered_df.pivot(index='feature', columns='model', values='importance')

    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(16, 10), color=BLUE_SHADES[:4], alpha=0.8)

    plt.xlabel('Features', fontsize=16, fontweight='bold')
    plt.ylabel('Permutation Importance', fontsize=16, fontweight='bold')
    plt.title('Feature Importance Comparison - Top 10 Features Across Models',
              fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Models')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/feature_importance_single.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved single feature importance comparison")


def plot_precision_recall_curves_enhanced(models, X_test, y_test):
    """Enhanced Precision-Recall curves with better styling"""
    plt.figure(figsize=(14, 10))

    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0.5, 0.5], 'k--', linewidth=2, label='Random Classifier', alpha=0.7)

    for i, (name, model) in enumerate(models.items()):
        color = BLUE_SHADES[i % len(BLUE_SHADES)]

        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)
                precision, recall, _ = precision_recall_curve(y_test, probas[:, 1])
                pr_auc = auc(recall, precision)

                plt.plot(recall, precision, color=color, linewidth=3,
                         label=f'{name} (AUC = {pr_auc:.3f})', alpha=0.8)

                # Add subtle fill under curve
                plt.fill_between(recall, precision, alpha=0.1, color=color)

            except Exception as e:
                print(f"⚠️ Error generating PR curve for {name}: {str(e)}")
                continue

    plt.xlabel('Recall (Sensitivity)', fontsize=16, fontweight='bold')
    plt.ylabel('Precision', fontsize=16, fontweight='bold')
    plt.title('Precision-Recall Curves - All Models', fontsize=18, fontweight='bold', pad=20)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.4, 1.05])

    plt.tight_layout()
    plt.savefig('output/precision_recall_curves_enhanced.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved enhanced precision-recall curves")


def plot_training_time_vs_accuracy_enhanced(results):
    """Enhanced training time vs accuracy using bar-line combination"""
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Sort by accuracy
    df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)

    # Bar positions
    x = np.arange(len(df))
    width = 0.6

    # Plot accuracy bars
    bars = ax1.bar(x, df['test_accuracy'] * 100, width,
                   color=BLUE_SHADES, alpha=0.8,
                   edgecolor='white', linewidth=2,
                   label='Test Accuracy')

    ax1.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold', color=DARK_BLUE)
    ax1.tick_params(axis='y', labelcolor=DARK_BLUE)
    ax1.set_ylim(85, 100)

    # Create second y-axis for training time
    ax2 = ax1.twinx()

    # Plot training time as line with markers
    line = ax2.plot(x, df['training_time'], 'o-', color='#08306b',
                    linewidth=3, markersize=10, markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor='#08306b',
                    label='Training Time')

    ax2.set_ylabel('Training Time (seconds)', fontsize=16, fontweight='bold', color='#08306b')
    ax2.tick_params(axis='y', labelcolor='#08306b')

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=14)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, fontweight='bold',
                     color=DARK_BLUE)

        # Add training time annotations
        time_val = df.iloc[i]['training_time']
        ax2.annotate(f'{time_val:.2f}s',
                     xy=(i, time_val),
                     xytext=(0, 10), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold',
                     color='#08306b')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Model Performance: Accuracy vs Training Time', fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/training_time_vs_accuracy_enhanced.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved enhanced training time vs accuracy plot")


def plot_performance_radar_enhanced(results):
    """Enhanced radar chart with blue theme"""
    fig = plt.figure(figsize=(14, 12))

    # Calculate additional metrics
    for result in results:
        precision = result['precision']
        recall = result['sensitivity']
        result['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # Add efficiency score (combination of accuracy and speed)
        result['efficiency'] = result['test_accuracy'] * (1 / (result['training_time'] + 0.1))

    # Metrics for radar chart
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Efficiency']
    categories = metrics
    N = len(categories)

    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Create polar subplot
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines and labels
    plt.xticks(angles[:-1], categories, size=13, fontweight='bold')

    # Draw y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.8, 1.0], ["0.6", "0.8", "1.0"], color="grey", size=11)
    plt.ylim(0.5, 1.0)

    # Plot each model
    for i, result in enumerate(results):
        values = [
            result['test_accuracy'],
            result['sensitivity'],
            result['specificity'],
            result['precision'],
            result['f1_score'],
            result['efficiency'] / max([r['efficiency'] for r in results])  # Normalize efficiency
        ]
        values += values[:1]  # Complete the circle

        # Use blue shades
        color = BLUE_SHADES[i % len(BLUE_SHADES)]

        ax.plot(angles, values, linewidth=2.5, linestyle='-',
                label=result['model'], color=color, alpha=0.9)
        ax.fill(angles, values, alpha=0.15, color=color)

    plt.title('Comprehensive Model Performance Radar Chart\n(All Metrics Normalized)',
              size=18, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True,
               fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('output/performance_radar_enhanced.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved enhanced performance radar chart")


def plot_model_ranking_comparison(results):
    """Horizontal bar chart showing model rankings across different metrics"""
    df = pd.DataFrame(results)

    # Calculate rankings for each metric
    metrics = ['test_accuracy', 'sensitivity', 'specificity', 'precision', 'training_time']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Speed (1/time)']

    # For training time, lower is better (inverse ranking)
    df['speed_score'] = 1 / df['training_time']
    df['test_accuracy_rank'] = df['test_accuracy'].rank(ascending=False)
    df['sensitivity_rank'] = df['sensitivity'].rank(ascending=False)
    df['specificity_rank'] = df['specificity'].rank(ascending=False)
    df['precision_rank'] = df['precision'].rank(ascending=False)
    df['speed_rank'] = df['speed_score'].rank(ascending=False)

    ranking_data = df[['model', 'test_accuracy_rank', 'sensitivity_rank',
                       'specificity_rank', 'precision_rank', 'speed_rank']]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(ranking_data))
    width = 0.15

    bars1 = ax.bar(x - 2 * width, ranking_data['test_accuracy_rank'], width, label='Accuracy', color='#1f77b4')
    bars2 = ax.bar(x - width, ranking_data['sensitivity_rank'], width, label='Sensitivity', color='#ff7f0e')
    bars3 = ax.bar(x, ranking_data['specificity_rank'], width, label='Specificity', color='#2ca02c')
    bars4 = ax.bar(x + width, ranking_data['precision_rank'], width, label='Precision', color='#d62728')
    bars5 = ax.bar(x + 2 * width, ranking_data['speed_rank'], width, label='Speed', color='#9467bd')

    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_title('Model Rankings Across Different Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranking_data['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/model_ranking_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved model ranking comparison")


def plot_error_analysis_comparison(results, models, X_test, y_test):
    """Compare error patterns across all models"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']

    error_data = []

    for i, (name, model) in enumerate(models.items()):
        if i >= 7:  # Only plot first 7 models
            break

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Calculate error rates
        total_errors = cm[0, 1] + cm[1, 0]  # FP + FN
        false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

        error_data.append({
            'model': name,
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'total_errors': total_errors,
            'fp_rate': false_positive_rate,
            'fn_rate': false_negative_rate
        })

    # Create error comparison bar chart
    error_df = pd.DataFrame(error_data)

    x = np.arange(len(error_df))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, error_df['false_positives'], width,
            label='False Positives', color='orange', alpha=0.7)
    plt.bar(x + width / 2, error_df['false_negatives'], width,
            label='False Negatives', color='red', alpha=0.7)

    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Errors', fontsize=14, fontweight='bold')
    plt.title('Error Analysis: False Positives vs False Negatives', fontsize=16, fontweight='bold')
    plt.xticks(x, error_df['model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/error_analysis_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print("✅ Saved error analysis comparison")




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



def save_all_visualizations(models, results, X_train, y_train, X_test, y_test):
    """Generate and save all visualizations"""
    print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS...")

    # Get feature names (assuming 30 features from WDBC dataset)
    feature_names = [f'Feature_{i + 1}' for i in range(X_test.shape[1])]

    # Generate all plots
    plot_comprehensive_performance(results)
    plot_comprehensive_learning_curves(models, X_train, y_train)
    plot_confusion_matrices_comprehensive(models, X_test, y_test)
    plot_roc_curves_comprehensive(models, X_test, y_test)

    # New enhanced visualizations with blue theme
    plot_comprehensive_learning_curves_single(models, X_train, y_train)
    plot_feature_importance_single(models, feature_names, X_test, y_test)
    plot_precision_recall_curves_enhanced(models, X_test, y_test)
    plot_training_time_vs_accuracy_enhanced(results)
    plot_performance_radar_enhanced(results)
    plot_model_ranking_comparison(results)
    plot_error_analysis_comparison(results, models, X_test, y_test)

    print("\n✅ ALL VISUALIZATIONS SAVED:")
    print("   - output/comprehensive_performance.png")
    print("   - output/comprehensive_learning_curves.png")
    print("   - output/confusion_matrices_comprehensive.png")
    print("   - output/roc_curves_comprehensive.png")
    print("   - output/learning_curves_single.png")
    print("   - output/feature_importance_single.png")
    print("   - output/precision_recall_curves_enhanced.png")
    print("   - output/training_time_vs_accuracy_enhanced.png")
    print("   - output/performance_radar_enhanced.png")
    print("   - output/model_ranking_comparison.png")
    print("   - output/error_analysis_comparison.png")