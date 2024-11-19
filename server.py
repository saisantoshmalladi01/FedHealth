import flwr as fl

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        train_accuracies = [res.metrics.get("train_accuracy", 0.0) for _, res in results]
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0.0
        print(f"Round {server_round} - Global Train Accuracy: {avg_train_accuracy:.4f}")
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        test_accuracies = [res.metrics.get("test_accuracy", 0.0) for _, res in results]
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else 0.0
        print(f"Round {server_round} - Global Test Accuracy: {avg_test_accuracy:.4f}")
        return aggregated_metrics

if __name__ == "__main__":
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),  # Increased rounds
        strategy=strategy,
    )
