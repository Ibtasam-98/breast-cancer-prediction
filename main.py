import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from preprocessing.preprocessing import load_data, feature_selection, prepare_data, scale_data
from models.model_definitions import get_models
from models.model_utils import evaluate_model
from visualization.plots import plot_metrics_comparison, plot_learning_curves, plot_confusion_matrices, \
    plot_feature_correlations, plot_feature_importances, plot_roc_curves
from config import settings


def save_models_and_scaler(models, scaler, folder='saved_models'):
    """Save trained models and scaler to disk"""
    os.makedirs(folder, exist_ok=True)

    # Save models
    for name, model in models.items():
        filename = os.path.join(folder, f"{name.lower().replace(' ', '_')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {filename}")

    # Save scaler
    scaler_filename = os.path.join(folder, 'scaler.pkl')
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler: {scaler_filename}")


def train_and_evaluate_models(model_definitions, X_train, y_train, X_test, y_test):
    """Helper function to train and evaluate models"""
    results = []
    trained_models = {}

    for name, definition in model_definitions.items():
        print(f"\nTraining {name}...")
        try:
            metrics = evaluate_model(
                definition['model'],
                definition['params'],
                definition.get('learning_rate_param'),
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                'model': name,
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'training_time': metrics['training_time'],
                'best_params': metrics['best_params'],
                'learning_rate': metrics.get('learning_rate'),
                'classification_report': metrics['classification_report']
            })

            trained_models[name] = metrics['model']
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    return results, trained_models


def print_results(results):
    """Helper function to print results"""
    if not results:
        print("No results to display")
        return

    results_df = pd.DataFrame(results)
    columns_to_display = ['model', 'train_accuracy', 'test_accuracy', 'training_time']
    if 'learning_rate' in results_df.columns:
        columns_to_display.append('learning_rate')
    print(results_df[columns_to_display].sort_values('test_accuracy', ascending=False).to_string(index=False))


def main():
    # Create output directories if they don't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('saved_models/all_features', exist_ok=True)  # Ensure this directory exists

    # Get model definitions first
    model_definitions = get_models()

    # Load and preprocess data
    try:
        data = load_data('dataset/breast-cancer.csv')
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    try:
        data, high_corr_features = feature_selection(data)
        print("\nFeature correlations with target:")
        print(data.corr()['diagnosis'].abs().sort_values(ascending=False))

        print("\nHighly correlated features selected:")
        print(high_corr_features)
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        return

    # Plot feature correlations if we have features
    if high_corr_features:
        try:
            # Ensure we only plot features that exist in the dataframe
            existing_features = [f for f in high_corr_features if f in data.columns]
            if existing_features:
                corr_fig = plot_feature_correlations(data, existing_features)
                corr_fig.savefig('output/feature_correlations.png', bbox_inches='tight')
                plt.close(corr_fig)
                print("\nSaved feature correlation plot to output/feature_correlations.png")
            else:
                print("\nNo existing features found for correlation plot")
        except Exception as e:
            print(f"Error plotting feature correlations: {str(e)}")
    else:
        print("\nNo highly correlated features found - skipping correlation plot")

    # Prepare data with all features
    try:
        X_train, X_test, y_train, y_test = prepare_data(data)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)  # Get the scaler object
    except Exception as e:
        print(f"Error preparing all-features data: {str(e)}")
        return

    # Prepare data with only highly correlated features (if any exist)
    results_hc = []
    models_hc = {}

    if high_corr_features:
        try:
            # Ensure we only use features that exist in the dataframe
            existing_features = [f for f in high_corr_features if f in data.columns]
            if existing_features:
                high_corr_data = data[existing_features + ['diagnosis']]
                X_train_hc, X_test_hc, y_train_hc, y_test_hc = prepare_data(high_corr_data)
                X_train_hc_scaled, X_test_hc_scaled, scaler_hc = scale_data(X_train_hc, X_test_hc)

                # Train and evaluate models on highly correlated features only
                print("\n=== Training on highly correlated features only ===")
                results_hc, models_hc = train_and_evaluate_models(
                    model_definitions,
                    X_train_hc_scaled,
                    y_train_hc,
                    X_test_hc_scaled,
                    y_test_hc
                )
                # Save models trained on highly correlated features
                save_models_and_scaler(models_hc, scaler_hc, 'saved_models/high_corr_features')
            else:
                print("\nNo existing highly correlated features - skipping this training")
        except Exception as e:
            print(f"Error processing highly correlated features: {str(e)}")
            results_hc = []
            models_hc = {}
    else:
        print("\nNo highly correlated features - skipping training on subset")

    # Train and evaluate models on all features
    print("\n=== Training on all features ===")
    results_all, models_all = train_and_evaluate_models(
        model_definitions,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    # Save models trained on all features AND the scaler
    save_models_and_scaler(models_all, scaler, 'saved_models/all_features')

    # Compare results
    print("\n=== Performance Comparison ===")
    print("\nAll features:")
    print_results(results_all)

    if results_hc:
        print("\nHighly correlated features only:")
        print_results(results_hc)
    else:
        print("\nNo results for highly correlated features")

    # Print detailed classification reports
    print("\n=== Detailed Classification Reports ===")
    for result in results_all:
        print(f"\n{result['model']}:")
        print(f"Best params: {result['best_params']}")
        if 'classification_report' in result:
            print("Classification Report:")
            print(result['classification_report'])
        else:
            print("No classification report available")
        print(f"Training time: {result['training_time']:.2f} seconds")
        print("-" * 80)

    # Generate all visualizations
    try:
        # Performance comparison plot (if we have both sets of results)
        if results_all and results_hc:
            comparison_fig = plot_metrics_comparison(results_all, results_hc)
            comparison_fig.savefig('output/performance_comparison.png', bbox_inches='tight')
            plt.close(comparison_fig)
            print("\nSaved performance comparison plot to output/performance_comparison.png")

        # Learning curves for all models
        learning_curve_fig = plot_learning_curves(models_all, X_train_scaled, y_train)
        learning_curve_fig.savefig('output/learning_curves_all.png', bbox_inches='tight')
        plt.close(learning_curve_fig)
        print("Saved learning curves plot to output/learning_curves_all.png")

        # Confusion matrices for all models
        confusion_matrix_fig = plot_confusion_matrices(models_all, X_test_scaled, y_test)
        confusion_matrix_fig.savefig('output/confusion_matrices_all.png', bbox_inches='tight')
        plt.close(confusion_matrix_fig)
        print("Saved confusion matrices plot to output/confusion_matrices_all.png")

        # Feature importance for tree-based models
        plot_feature_importances(models_all, X_train.columns)

        plot_roc_curves(models_all, X_test_scaled, y_test)

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")


if __name__ == "__main__":
    main()