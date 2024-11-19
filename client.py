import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from utils import TabularModel, get_client_data
import argparse
import numpy as np


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data, device):
        self.model = model.to(device)
        self.train_features, self.train_labels = train_data
        self.test_features, self.test_labels = test_data
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=10, mode="triangular"
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        best_loss = float("inf")
        patience, patience_counter = 5, 0

        for epoch in range(100):  # Train for up to 100 epochs
            outputs = self.model(self.train_features).squeeze()
            loss = self.criterion(outputs, self.train_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        with torch.no_grad():
            outputs = self.model(self.train_features).squeeze()
            predicted = (outputs > 0.5).float()
            train_accuracy = (predicted == self.train_labels).sum().item() / len(self.train_labels)
        print(f"Training complete - Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}")
        return self.get_parameters(config), len(self.train_features), {"train_accuracy": train_accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_features).squeeze()
            loss = self.criterion(outputs, self.test_labels).item()
            predicted = (outputs > 0.5).float()
            test_accuracy = (predicted == self.test_labels).sum().item() / len(self.test_labels)

            # Save true labels and predictions for confusion matrix
            true_labels = self.test_labels.cpu().numpy()
            predicted_labels = predicted.cpu().numpy()
            np.save("true_labels.npy", true_labels)
            np.save("predicted_labels.npy", predicted_labels)

        print(f"Evaluation complete - Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return loss, len(self.test_features), {"test_accuracy": test_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    client_id = args.client_id
    train_features, test_features, train_labels, test_labels = get_client_data(client_id, num_clients=3)
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    model = TabularModel(input_dim=train_features.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = FlowerClient(model, (train_features, train_labels), (test_features, test_labels), device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
