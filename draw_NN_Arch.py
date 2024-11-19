import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the layers in your neural network
layers = [
    ("Input Layer", 8),  # Example: 8 input features
    ("Dense (256)", 256),
    ("ReLU", 256),
    ("Dropout (40%)", 256),
    ("Dense (128)", 128),
    ("ReLU", 128),
    ("Dropout (30%)", 128),
    ("Dense (64)", 64),
    ("ReLU", 64),
    ("Dropout (20%)", 64),
    ("Output Layer (Sigmoid)", 1)
]

# Plot settings
fig, ax = plt.subplots(figsize=(8, len(layers) * 1.5))
ax.set_xlim(0, 3)
ax.set_ylim(-len(layers), 1)

# Draw the layers
for i, (layer_name, size) in enumerate(layers):
    # Add a rectangle for each layer
    rect = Rectangle((1, -i - 1), 1, 0.8, edgecolor="black", facecolor="skyblue")
    ax.add_patch(rect)
    
    # Add text for layer name and size
    ax.text(1.5, -i - 0.6, f"{layer_name}\n(Size: {size})", ha="center", va="center", fontsize=10)

# Final adjustments
ax.set_axis_off()
plt.title("Neural Network Architecture (Block Diagram)", fontsize=14)
plt.show()
