#client.py
import flwr as fl
import numpy as np
from model import create_model

class SmartGridClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, optimizer_type="yogi"):
        self.model = create_model(x_train.shape[1], optimizer_type=optimizer_type)
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.y_train = y_train

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # ---- Adaptive Batch Size Computation ----
        Ri = np.random.uniform(1.0, 5.0)
        alpha = 0.5
        Bmax = 64
        Bt = min(Bmax, int(Ri ** alpha * 16))

        print(f"[Client] Adaptive Batch Size (Bt): {Bt:.0f} based on Ri={Ri:.2f}")
        # -----------------------------------------

        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=Bt, verbose=0)

        # ---- Model Compression & Privacy ----
        original_weights = self.model.get_weights()
        theta = 0.01
        compressed_weights = [np.sign(w) * np.maximum(theta, np.abs(w)) for w in original_weights]
        sigma = 0.001
        noisy_weights = [w + np.random.normal(0, sigma, size=w.shape) for w in compressed_weights]
        # --------------------------------------

        return noisy_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"mae": mae}
