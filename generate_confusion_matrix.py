import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load true and predicted labels
true_labels = np.load("true_labels.npy")
predicted_labels = np.load("predicted_labels.npy")

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title("Federated Neural Network - Confusion Matrix")
plt.show()
