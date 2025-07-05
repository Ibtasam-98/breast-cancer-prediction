import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_feature_correlations(data, high_corr_features):
    """Plot enhanced correlation matrix of highly correlated features with light colors"""
    # Calculate correlations
    corr = data[high_corr_features + ['diagnosis']].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot heatmap with lighter colors
    fig, ax = plt.subplots(figsize=(14, 12))
    # Using a light color palette - 'icefire' is good for correlations (blue-red)
    # Alternatively, 'coolwarm' is another good option that's lighter
    cmap = sns.light_palette("#5B9BD5", as_cmap=True)  # Light blue palette

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', square=True,
                linewidths=.5, cbar_kws={'shrink': .5},
                vmin=-1, vmax=1)  # Explicitly set range for correlations

    plt.title('Feature Correlations with Diagnosis', pad=20)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    return fig



def plot_metrics_comparison(results_all, results_hc):
    """Plot comparison between all features and highly correlated features results"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.suptitle('Model Performance: All Features vs Highly Correlated Features', y=1.05, fontsize=16)

    # Convert results to DataFrames
    df_all = pd.DataFrame(results_all)
    df_hc = pd.DataFrame(results_hc)

    # Plot test accuracy comparison
    sns.barplot(x='test_accuracy', y='model', data=df_all.sort_values('test_accuracy', ascending=False),
                palette='Blues', ax=axes[0])
    axes[0].set_title('All Features', pad=20)
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_ylabel('Model')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    sns.barplot(x='test_accuracy', y='model', data=df_hc.sort_values('test_accuracy', ascending=False),
                palette='Greens', ax=axes[1])
    axes[1].set_title('Highly Correlated Features Only', pad=20)
    axes[1].set_xlabel('Test Accuracy')
    axes[1].set_ylabel('')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    plt.tight_layout()
    return fig


def plot_learning_curves(models, X_train, y_train):
    """Plot learning curves for all models in a single visualization"""
    plt.figure(figsize=(12, 8))
    plt.title('Learning Curves Comparison', pad=20, fontsize=16)

    # Define a color palette
    colors = sns.color_palette('husl', n_colors=len(models))

    for (name, model), color in zip(models.items(), colors):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot training accuracy
        plt.plot(train_sizes, train_mean, 'o-', color=color,
                 label=f'{name} (Train)')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color=color)

        # Plot validation accuracy
        plt.plot(train_sizes, test_mean, 's--', color=color,
                 label=f'{name} (Val)')
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color=color)

    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    plt.tight_layout()
    fig = plt.gcf()
    return fig



def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(models)
    rows = (n_models + 1) // 2
    fig = plt.figure(figsize=(20, rows * 8))
    plt.suptitle('Confusion Matrices', y=1.02, fontsize=16)

    for i, (name, model) in enumerate(models.items(), 1):
        ax = plt.subplot(rows, 2, i)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'],
                    cbar=False)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j + 0.5, i + 0.25, f"{cm[i, j]}",
                        ha='center', va='center', color='black')

        ax.set_title(f'{name}', pad=20)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    return fig


def plot_feature_importances(models, feature_names):
    """Plot feature importances for tree-based models"""
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

        # Use feature names if available, otherwise use indices
        if feature_names is not None:
            labels = np.array(feature_names)[indices]
        else:
            labels = indices

        # Plot
        ax.set_title(f"Feature Importances - {name}")
        ax.bar(range(len(importances)), importances[indices], align="center")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim([-1, len(importances)])
        ax.set_ylabel("Importance")

    plt.tight_layout()
    fig.savefig('output/feature_importances.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved feature importances plot to output/feature_importances.png")




def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    y_test_bin = label_binarize(y_test, classes=[0, 1])

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test_bin, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('output/roc_curves.png', bbox_inches='tight')
    plt.close()
    print("Saved ROC curves plot to output/roc_curves.png")


def plot_accuracy_comparison(results):
    """
    Plot comparison of training and testing accuracy for all models

    Args:
        results: List of dictionaries containing model results
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Sort by test accuracy
    df = df.sort_values('test_accuracy', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set positions and width for bars
    x = np.arange(len(df))
    width = 0.35

    # Plot bars
    train_bars = ax.bar(x - width / 2, df['train_accuracy'], width,
                        label='Training Accuracy', color='navy', alpha=0.7)
    test_bars = ax.bar(x + width / 2, df['test_accuracy'], width,
                       label='Testing Accuracy', color='darkorange', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Testing Accuracy Across Models', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_labels(train_bars)
    add_labels(test_bars)

    # Calculate and display the gap between train and test accuracy
    for i, (train_acc, test_acc) in enumerate(zip(df['train_accuracy'], df['test_accuracy'])):
        gap = train_acc - test_acc
        ax.text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.1%}',
                ha='center', va='bottom', fontsize=8, color='red')

    # Adjust layout
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    return fig