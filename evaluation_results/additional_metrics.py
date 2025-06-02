import numpy as np
import pandas as pd

def compute_metrics_from_confusion(conf_mat):
    num_classes = conf_mat.shape[0]
    TP = np.diag(conf_mat)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP
    TN = conf_mat.sum() - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    class_accuracy = TP / (TP + FP + FN + 1e-8)  # Optional

    global_accuracy = TP.sum() / conf_mat.sum()
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    macro_dice = dice.mean()

    results = {
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dice": dice,
            "accuracy": class_accuracy
        },
        "global": {
            "accuracy": global_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_dice": macro_dice
        }
    }
    return results

# Read confusion matrix from CSV
conf_df = pd.read_csv('confusion_matrix_v2.csv', index_col=0)
conf_mat_total = conf_df.values

results = compute_metrics_from_confusion(conf_mat_total)

# Print per-class metrics
for i in range(len(conf_mat_total)):
    print(f"Class {i}: "
          f"Precision={results['per_class']['precision'][i]:.4f}, "
          f"Recall={results['per_class']['recall'][i]:.4f}, "
          f"F1={results['per_class']['f1'][i]:.4f}, "
          f"Dice={results['per_class']['dice'][i]:.4f}")

# Print global metrics
print("\nGlobal Accuracy:", results['global']['accuracy'])
print("Macro Precision:", results['global']['macro_precision'])
print("Macro Recall:", results['global']['macro_recall'])
print("Macro F1 Score:", results['global']['macro_f1'])
print("Macro Dice Coefficient:", results['global']['macro_dice'])

