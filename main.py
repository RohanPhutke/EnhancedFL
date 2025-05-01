# smart_grid_federated/main.py
import numpy as np
import threading
import flwr as fl
from client import SmartGridClient
from utils import load_and_preprocess_data, split_data_among_clients, load_test_data
from model import create_model  # Assuming you have a model builder
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

if __name__ == "__main__":
    # Load and preprocess data
    X_scaled, y, _ = load_and_preprocess_data("data/smart_grid_dataset.csv")
    client_splits = split_data_among_clients(X_scaled, y, num_clients=5)

    def client_fn(X_c, y_c):
        client = SmartGridClient(np.array(X_c), np.array(y_c), optimizer_type="yogi")
        logging.info("Starting client with %d samples", len(X_c))
        fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    threads = []
    for X_c, y_c in client_splits:
        thread = threading.Thread(target=client_fn, args=(X_c, y_c))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    logging.info("All clients have finished training.")

    # Final global model evaluation using saved server weights
    logging.info("Evaluating final global model...")
    x_test, y_test = load_test_data("data/smart_grid_dataset.csv")
    model = create_model(x_test.shape[1],optimizer_type="yogi")  # Uses default optimizer_type from model.py (yogi)

    try:
        with open("best_model_weights.pkl", "rb") as f:
            global_weights = pickle.load(f)
        model.set_weights(global_weights)
        logging.info("✅ Loaded best global model weights for evaluation.")
    except FileNotFoundError:
        logging.warning("⚠️ Global weights file not found. Evaluating untrained model.")

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nFinal Evaluation on Central Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")