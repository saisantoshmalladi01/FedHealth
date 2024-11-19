import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Define the TabularModel class
class TabularModel(nn.Module):
    def __init__(self, input_dim):
        super(TabularModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# General function for training and evaluation
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=50, patience=5):
    model.to(device)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_loss /= len(train_loader)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            all_predictions, all_labels = [], []
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                all_predictions.extend((outputs > 0.5).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = accuracy_score(all_labels, all_predictions)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Main function
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('./diabetes_prediction_dataset.csv')

    # Map and preprocess categorical columns
    df['smoking_history'] = df['smoking_history'].map({
        'No Info': 0, 'never': 0, 'former': 1, 'current': 1, 'not current': 1, 'ever': 1
    })
    df['gender'] = df['gender'].map({'Female': 0, 'Other': 0, 'Male': 1})
    df.dropna(inplace=True)

    # Prepare features and labels
    X = df.drop(columns=["diabetes"]).values.astype('float32')
    y = df["diabetes"].values.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = TabularModel(input_dim=X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=10, mode="triangular")

    # Train and evaluate the model
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, device)
