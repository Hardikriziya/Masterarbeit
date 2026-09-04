import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

result_folder = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Results\results_LSTM_AllCon\SDU\C4"      


# Load prediction files
y_true = np.load(os.path.join(result_folder, "clf_true.npy"))
y_pred = np.load(os.path.join(result_folder, "clf_pred.npy"))

# Class labels
labels = [
    "RUL>400",
    "RUL>300",
    "RUL>200",
    "RUL>100",
    "RUL<100"
]

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    target_names=labels,
    digits=4
)

cm = confusion_matrix(y_true, y_pred)


# Display Results
print("="*60)
print(f"Test Accuracy : {accuracy:.4f}")
print("="*60)

print("\nClassification Report\n")
print(report)

print("\nConfusion Matrix\n")
print(cm)


# Save Report
report_path = os.path.join(result_folder, "classification_results.txt")

with open(report_path, "w") as f:
    f.write("="*60 + "\n")
    f.write(f"Test Accuracy : {accuracy:.4f}\n")
    f.write("="*60 + "\n\n")

    f.write("Classification Report\n")
    f.write("---------------------\n")
    f.write(report)

    f.write("\n\nConfusion Matrix\n")
    f.write("----------------\n")
    f.write(np.array2string(cm))

print("\nClassification report saved to:")
print(report_path)


# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()


# Save Figure
figure_path = os.path.join(result_folder, "confusion_matrix_Selected_1000.png")

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved to:")
print(figure_path)
plt.show()