import pandas as pd
import numpy as np

conf_df = pd.read_csv('confusion_matrix.csv', index_col=0)
conf_mat_total = conf_df.values

# Binary class mapping:
# Class 0 = No Defect
# Classes 1–4 = Defect

TN = conf_mat_total[0, 0]
FP = conf_mat_total[0, 1:].sum()
FN = conf_mat_total[1:, 0].sum()
TP = conf_mat_total[1:, 1:].sum()

binary_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
binary_dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

print(f"\nBinary Accuracy (Defect vs. No Defect): {binary_acc:.4f}")
print(f"Binary Dice Coefficient (Defect vs. No Defect): {binary_dice:.4f}")

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

print(f"Precision (Defect): {precision:.4f}")
print(f"Recall (Defect): {recall:.4f}")