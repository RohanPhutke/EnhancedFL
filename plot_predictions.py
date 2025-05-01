# plot_predictions.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from utils import load_test_data

def plot_predictions():
    # Load test data
    X_test, y_test = load_test_data("data/smart_grid_dataset.csv")

    # Load global model parameters
    with open("BEST_MSE-0.1864_R2-0.9792.pkl", "rb") as f:
        parameters = pickle.load(f)

    # Recreate model and set weights
    model = create_model(X_test.shape[1])
    model.set_weights(parameters)

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Sample 100 points for a cleaner plot
    sample_size = 100
    y_test_sample = y_test[:sample_size]
    y_pred_sample = y_pred[:sample_size]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sample, label="Actual", marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred_sample, label="Predicted", marker='x', linestyle='--', alpha=0.7)
    plt.title("Actual vs Predicted Load (kW) [Sampled]")
    plt.xlabel("Sample Index")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")  # Save to file
    plt.show()

if __name__ == "__main__":
    plot_predictions()
