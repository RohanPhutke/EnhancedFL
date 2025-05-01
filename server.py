#server.py
import flwr as fl
from model import create_model
from utils import load_test_data
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np
from flwr.common import ndarrays_to_parameters
from typing import List

# Global variables to track models
latest_global_parameters = None
best_model_parameters = None
best_r2 = float("-inf")  # Use float("inf") if you want to track by lowest MSE


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    mse_list = [m.get("mse", 0.0) * num_examples for num_examples, m in metrics]
    r2_list = [m.get("r2", 0.0) * num_examples for num_examples, m in metrics]
    return {
        "mse": sum(mse_list) / total_examples if total_examples > 0 else 0.0,
        "r2": sum(r2_list) / total_examples if total_examples > 0 else 0.0,
    }


def get_evaluate_fn():
    X_test, y_test = load_test_data("data/smart_grid_dataset.csv")

    def evaluate(server_round, parameters, config):
        global latest_global_parameters, best_model_parameters, best_r2

        latest_global_parameters = parameters

        model = create_model(X_test.shape[1])
        model.set_weights(parameters)

        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        loss = mse

        print(f"[Round {server_round}] Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save best model based on R²
        if r2 > best_r2:
            best_r2 = r2
            best_model_parameters = parameters
            print(f"🌟 New best model found at round {server_round} with R² = {r2:.4f}")

        return loss, {"mse": mse, "r2": r2}

    return evaluate


def start_server():
    # Initialize model and get initial parameters
    X_test, _ = load_test_data("data/smart_grid_dataset.csv")
    dummy_model = create_model(X_test.shape[1])
    initial_weights = dummy_model.get_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)
    # with open("best_model_weights.pkl", "rb") as f:
    #         new_weights = pickle.load(f)
    # initial_parameters = ndarrays_to_parameters(new_weights)

    strategy = fl.server.strategy.FedAdam(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        evaluate_fn=get_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=True,
        initial_parameters=initial_parameters,
    )

    # Start the federated server
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy
    )

    # # Save final global model weights
    # if latest_global_parameters is not None:
    #     with open("global_model_weights.pkl", "wb") as f:
    #         pickle.dump(latest_global_parameters, f)
    #     print("✅ Saved final global model weights to 'global_model_weights.pkl'")
    # else:
    #     print("⚠️ Could not save final weights: latest_global_parameters is None")

    # Save best model weights
    if best_model_parameters is not None:
        with open("best_model_weights.pkl", "wb") as f:
            pickle.dump(best_model_parameters, f)
        print("🌟 Saved best performing model weights to 'best_model_weights.pkl'")
    else:
        print("⚠️ Could not save best model: best_model_parameters is None")

    # Save training history
    try:
        extracted_history = {
            "loss_distributed": history.losses_distributed,
            "loss_centralized": history.losses_centralized,
            "metrics_distributed": history.metrics_distributed,
            "metrics_centralized": history.metrics_centralized,
        }
        with open("history.pkl", "wb") as f:
            pickle.dump(extracted_history, f)
        print("📈 Saved training history to 'history.pkl'")
    except Exception as e:
        print(f"⚠️ Could not save training history: {e}")


if __name__ == "__main__":
    start_server()
