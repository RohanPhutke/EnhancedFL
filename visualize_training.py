# visualize_training.py
import pickle
import matplotlib.pyplot as plt
import os

def load_history(filename="history.pkl"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"'{filename}' not found. Ensure it's saved after training.")
    with open(filename, "rb") as f:
        return pickle.load(f)

def plot_metrics(history):
    # Distributed Loss
    rounds_distributed = list(range(1, len(history["loss_distributed"]) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_distributed, history["loss_distributed"], label="Distributed Loss", marker='o')

    # Centralized Loss
    rounds_centralized, loss_centralized = zip(*history["loss_centralized"])
    plt.plot(rounds_centralized, loss_centralized, label="Centralized Loss", marker='s')

    plt.title("Loss Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # MSE
    rounds_mse, mse = zip(*history["metrics_centralized"]["mse"])
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_mse, mse, label="Centralized MSE", marker='o', color='orange')
    plt.title("MSE Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # R²
    rounds_r2, r2_scores = zip(*history["metrics_centralized"]["r2"])
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_r2, r2_scores, label="Centralized R² Score", marker='o', color='green')
    plt.title("R² Score Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history = load_history("history_BEST_MSE-0.1864_R2-0.9792.pkl")  # replace if different
    plot_metrics(history)
