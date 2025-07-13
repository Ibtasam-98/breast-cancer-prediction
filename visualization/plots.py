import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Define blue color palette
BLUE_PALETTE = sns.light_palette("#1f77b4", as_cmap=True)
BLUE_COLORS = sns.light_palette("#1f77b4", n_colors=10)
DARK_BLUE = "#1f77b4"
LIGHT_BLUE = "#aec7e8"


def plot_feature_correlations(data, high_corr_features):
    """Plot enhanced correlation matrix with blue theme"""
    corr = data[high_corr_features + ['diagnosis']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, cmap=BLUE_PALETTE, center=0,
                annot=True, fmt='.2f', square=True,
                linewidths=.5, cbar_kws={'shrink': .5},
                vmin=-1, vmax=1)

    plt.title('Feature Correlations with Diagnosis (Blue Theme)', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    fig.savefig('output/feature_correlations.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved feature correlations plot to output/feature_correlations.png")
    return fig


def plot_metrics_comparison(results_all, results_hc):
    """Plot comparison with blue theme"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.suptitle('Model Performance Comparison ', y=1.05, fontsize=16)

    df_all = pd.DataFrame(results_all)
    df_hc = pd.DataFrame(results_hc)

    # Use different blue shades for each plot
    sns.barplot(x='test_accuracy', y='model',
                data=df_all.sort_values('test_accuracy', ascending=False),
                palette=sns.light_palette(DARK_BLUE, n_colors=len(df_all)),
                ax=axes[0])
    axes[0].set_title('All Features', pad=20)
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_ylabel('Model')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    sns.barplot(x='test_accuracy', y='model',
                data=df_hc.sort_values('test_accuracy', ascending=False),
                palette=sns.light_palette(LIGHT_BLUE, n_colors=len(df_hc)),
                ax=axes[1])
    axes[1].set_title('Highly Correlated Features Only', pad=20)
    axes[1].set_xlabel('Test Accuracy')
    axes[1].set_ylabel('')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    plt.tight_layout()

    # Save figure
    fig.savefig('output/metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved metrics comparison plot to output/metrics_comparison.png")
    return fig


def plot_learning_curves(models, X_train, y_train):
    """Plot learning curves with blue theme"""
    plt.figure(figsize=(12, 8))
    plt.title('Learning Curves Comparison ', pad=20, fontsize=16)

    # Generate different blue shades
    blues = sns.light_palette(DARK_BLUE, n_colors=len(models) * 2)

    for i, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot with blue shades
        plt.plot(train_sizes, train_mean, 'o-', color=blues[i * 2],
                 label=f'{name} (Train)')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color=blues[i * 2])

        plt.plot(train_sizes, test_mean, 's--', color=blues[i * 2 + 1],
                 label=f'{name} (Val)')
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color=blues[i * 2 + 1])

    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    fig.savefig('output/learning_curves.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved learning curves plot to output/learning_curves.png")
    return fig


def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices with blue theme"""
    n_models = len(models)
    rows = (n_models + 1) // 2
    fig = plt.figure(figsize=(20, rows * 8))
    plt.suptitle('Confusion Matrices ', y=1.02, fontsize=16)

    for i, (name, model) in enumerate(models.items(), 1):
        ax = plt.subplot(rows, 2, i)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=BLUE_PALETTE,
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'],
                    cbar=False, vmin=0, vmax=1)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j + 0.5, i + 0.25, f"{cm[i, j]}",
                        ha='center', va='center', color='black')

        ax.set_title(f'{name}', pad=20)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()

    # Save figure
    fig.savefig('output/confusion_matrices.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved confusion matrices plot to output/confusion_matrices.png")
    return fig


def plot_feature_importances(models, feature_names):
    """Plot feature importances with blue theme"""
    tree_models = {
        name: model for name, model in models.items()
        if hasattr(model, 'feature_importances_')
    }

    if not tree_models:
        print("No tree-based models found for feature importance plots")
        return

    n_models = len(tree_models)
    fig = plt.figure(figsize=(20, 5 * n_models))

    for i, (name, model) in enumerate(tree_models.items(), 1):
        ax = plt.subplot(n_models, 1, i)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        labels = np.array(feature_names)[indices] if feature_names is not None else indices

        ax.set_title(f"Feature Importances - {name} (Blue Theme)")
        ax.bar(range(len(importances)), importances[indices],
               align="center", color=DARK_BLUE)
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim([-1, len(importances)])
        ax.set_ylabel("Importance")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig.savefig('output/feature_importances.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved feature importances plot to output/feature_importances.png")
    return fig


def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves with blue theme"""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    y_test_bin = label_binarize(y_test, classes=[0, 1])

    # Generate different blue shades
    blues = sns.light_palette(DARK_BLUE, n_colors=len(models))

    for (name, model), color in zip(models.items(), blues):
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test_bin, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f'{name} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves ')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    fig.savefig('output/roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved ROC curves plot to output/roc_curves.png")
    return fig


def plot_accuracy_comparison(results):
    """Plot accuracy comparison with blue theme"""
    df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(df))
    width = 0.35

    train_bars = ax.bar(x - width / 2, df['train_accuracy'], width,
                        label='Training Accuracy', color=DARK_BLUE, alpha=0.8)
    test_bars = ax.bar(x + width / 2, df['test_accuracy'], width,
                       label='Testing Accuracy', color=LIGHT_BLUE, alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Testing Accuracy', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    # Add value labels
    for bars in [train_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Add gap annotation in blue
    for i, (train_acc, test_acc) in enumerate(zip(df['train_accuracy'], df['test_accuracy'])):
        gap = train_acc - test_acc
        ax.text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.1%}',
                ha='center', va='bottom', fontsize=8, color=DARK_BLUE)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Save figure
    fig.savefig('output/accuracy_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved accuracy comparison plot to output/accuracy_comparison.png")
    return fig


def save_all_visualizations(models, data, feature_names, results_all, results_hc,
                            X_train, y_train, X_test, y_test, high_corr_features):
    """Generate and save all visualizations"""
    print("\nGenerating all visualizations ...")

    # Generate all plots
    plot_feature_correlations(data, high_corr_features)
    plot_metrics_comparison(results_all, results_hc)
    plot_learning_curves(models, X_train, y_train)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importances(models, feature_names)
    plot_roc_curves(models, X_test, y_test)
    plot_accuracy_comparison(results_all)

    print("\nAll visualizations saved to the 'output' directory")