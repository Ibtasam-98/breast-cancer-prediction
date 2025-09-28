import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from preprocessing.preprocessing import load_data, prepare_data_enhanced, scale_data_enhanced
from models.model_definitions import get_models
from models.model_utils import evaluate_model_enhanced
from visualization.plots import save_all_visualizations
from config import settings


def save_models_and_scaler(models, scaler, folder='saved_models'):
    """Save trained models and scaler to disk"""
    os.makedirs(folder, exist_ok=True)

    # Save models
    for name, model in models.items():
        filename = os.path.join(folder, f"{name.lower().replace(' ', '_')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Saved model: {filename}")

    # Save scaler
    scaler_filename = os.path.join(folder, 'scaler.pkl')
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Saved scaler: {scaler_filename}")


def train_and_evaluate_selected_models(model_definitions, X_train, y_train, X_test, y_test):
    """Train and evaluate only the selected models with detailed output"""
    results = []
    trained_models = {}

    for name, definition in model_definitions.items():
        try:
            metrics = evaluate_model_enhanced(
                definition['model'],
                definition['params'],
                definition.get('learning_rate_param'),
                X_train,
                y_train,
                X_test,
                y_test,
                name
            )

            results.append({
                'model': name,
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'training_time': metrics['training_time'],
                'best_params': metrics['best_params'],
                'learning_rate': metrics.get('learning_rate'),
                'classification_report': metrics['classification_report'],
                'confusion_matrix': metrics['confusion_matrix'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision']
            })

            trained_models[name] = metrics['model']

        except Exception as e:
            print(f"❌ Error training {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return results, trained_models


def print_final_summary(results):
    """Print comprehensive final summary"""
    print("\n" + "=" * 80)
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 80)

    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model'],
            'Train Acc': result['train_accuracy'],
            'Test Acc': result['test_accuracy'],
            'Time (s)': result['training_time'],
            'Sensitivity': result['sensitivity'],
            'Specificity': result['specificity'],
            'Precision': result['precision']
        })

    df = pd.DataFrame(summary_data)
    df['Rank'] = df['Test Acc'].rank(method='dense', ascending=False).astype(int)
    df = df.sort_values('Rank')

    # Format the dataframe for nice printing
    display_df = df.copy()
    display_df['Train Acc'] = display_df['Train Acc'].apply(lambda x: f'{x:.4f}')
    display_df['Test Acc'] = display_df['Test Acc'].apply(lambda x: f'{x:.4f}')
    display_df['Time (s)'] = display_df['Time (s)'].apply(lambda x: f'{x:.2f}')
    display_df['Sensitivity'] = display_df['Sensitivity'].apply(lambda x: f'{x:.4f}')
    display_df['Specificity'] = display_df['Specificity'].apply(lambda x: f'{x:.4f}')
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.4f}')

    print("\n📊 PERFORMANCE RANKING:")
    print(display_df.to_string(index=False))

    # Find best model
    best_model = max(results, key=lambda x: x['test_accuracy'])
    worst_model = min(results, key=lambda x: x['test_accuracy'])

    print(f"\n🎯 BEST PERFORMER: {best_model['model']}")
    print(f"   Test Accuracy: {best_model['test_accuracy']:.4f} ({best_model['test_accuracy'] * 100:.2f}%)")
    print(f"   Training Time: {best_model['training_time']:.2f}s")

    print(f"\n📉 WORST PERFORMER: {worst_model['model']}")
    print(f"   Test Accuracy: {worst_model['test_accuracy']:.4f} ({worst_model['test_accuracy'] * 100:.2f}%)")

    # Calculate statistics
    test_accuracies = [r['test_accuracy'] for r in results]
    training_times = [r['training_time'] for r in results]

    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Average Test Accuracy: {np.mean(test_accuracies):.4f} ({np.mean(test_accuracies) * 100:.2f}%)")
    print(f"   Std Test Accuracy:     {np.std(test_accuracies):.4f}")
    print(f"   Average Training Time: {np.mean(training_times):.2f}s")
    print(f"   Total Training Time:   {np.sum(training_times):.2f}s")

    print(f"\n🏅 TOP 3 MODELS:")
    top_3 = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)[:3]
    for i, model in enumerate(top_3, 1):
        print(f"   {i}. {model['model']}: {model['test_accuracy']:.4f}")

    print("=" * 80)


def main():
    # Create all necessary directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)

    # Set better plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    print("=" * 80)
    print("🧠 BREAST CANCER PREDICTION - SELECTED MODELS ONLY")
    print("=" * 80)
    print("SELECTED MODELS: XGBoost, Random Forest Optimized, Neural Network,")
    print("                 AdaBoost, SVM RBF Optimized, Stacking Enhanced,")
    print("                 SGD Classifier")
    print("=" * 80)

    # Get selected model definitions
    model_definitions = get_models()
    print(f"\n📋 LOADED {len(model_definitions)} MODELS:")
    for i, model_name in enumerate(model_definitions.keys(), 1):
        print(f"   {i}. {model_name}")

    # Load and preprocess data
    try:
        print("\n📊 LOADING DATASET...")
        data = load_data('dataset/breast-cancer.csv')
        print(f"   Dataset shape: {data.shape}")
        print(f"   Features: {len(data.columns) - 1}")
        print(f"   Target distribution:")
        print(f"     Benign (0): {len(data[data['diagnosis'] == 0])}")
        print(f"     Malignant (1): {len(data[data['diagnosis'] == 1])}")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return

    # Prepare data
    try:
        print("\n🔄 PREPARING DATA...")
        X_train, X_test, y_train, y_test = prepare_data_enhanced(data)
        print(f"   Training set: {X_train.shape}")
        print(f"   Testing set:  {X_test.shape}")

        # Use standard scaling
        X_train_scaled, X_test_scaled, scaler = scale_data_enhanced(X_train, X_test, method='standard')
        print("   ✅ Data scaled using StandardScaler")

    except Exception as e:
        print(f"❌ Error preparing data: {str(e)}")
        return

    # Train selected models
    print("\n" + "=" * 80)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 80)

    results, trained_models = train_and_evaluate_selected_models(
        model_definitions,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    # Save models
    save_models_and_scaler(trained_models, scaler, 'saved_models')

    # Print final summary
    print_final_summary(results)

    # Generate visualizations
    try:
        save_all_visualizations(trained_models, results, X_train_scaled, y_train, X_test_scaled, y_test)
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("📁 Outputs saved in:")
    print("   - Terminal: Detailed training logs and performance metrics")
    print("   - output/: Comprehensive visualizations")
    print("   - saved_models/: Trained models and scaler")
    print("=" * 80)


if __name__ == "__main__":
    main()


#
#
# \begin{table*}[t]
# \centering
# \caption{Comparative Analysis of Breast Cancer Prediction Performance}
# \label{tab:comparison}
# \begin{tabular}{p{3.2cm}ccp{2.5cm}}
# \toprule
# \textbf{Study / Model} & \textbf{Accuracy (\%)} & \textbf{Best Model} & \textbf{Dataset} \\
# \midrule
# \textbf{Our Study} & & & \\
# \quad SGD Classifier & \textbf{98.25} & SGD & WDBC \\
# \quad XGBoost & \textbf{97.37} & XGBoost & WDBC \\
# \quad AdaBoost & \textbf{97.37} & AdaBoost & WDBC \\
# \quad SVM RBF Optimized & \textbf{97.37} & SVM & WDBC \\
# \quad Random Forest Optimized & \textbf{96.49} & RF & WDBC \\
# \quad Stacking Enhanced & \textbf{96.49} & Ensemble & WDBC \\
# \quad Neural Network & \textbf{93.86} & NN & WDBC \\
# \midrule
# \textbf{Literature Review} & & & \\
# \quad Rawal and Ramik (2020) & 97.10 & Random Forest & WBCD \\
# \quad Chen et al. (2023) & 97.20 & XGBoost & WDBC \\
# \quad La et al. (2025) & 91.67 & Logistic Regression & Clinical Data \\
# \quad Naji et al. (2021) & 96.80 & SVM & WDBC \\
# \quad Sumbaly et al. (2014) & 93.56 & Decision Tree (J48) & WBCD \\
# \quad Li and Chen (2018) & 96.50 & Random Forest & WBCD+BCCD \\
# \quad Banu et al. (2025) & 86.34 & SVM & Mammography \\
# \quad Kavitha et al. (2025) & 96.00 & YOLOv3 & Ultrasound \\
# \quad Tanveer et al. (2025) & 95.80 & CNN & MIAS+DDSM \\
# \bottomrule
# \end{tabular}
# \end{table*}