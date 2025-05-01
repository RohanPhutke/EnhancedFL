# 🧠 **Goal of the Project**

We're building a system that predicts the **"Predicted Load (kW)"** of a smart grid using **Federated Learning (FL)** — meaning multiple clients train on local data and share only model weights with a central server, ensuring privacy and data locality.

---

## 🔁 **Methodology Overview**

### 1. **Data Loading & Preprocessing**  
From `utils.py`:  
- Load smart grid CSV data
- Drop timestamps
- Scale features using `StandardScaler`
- Split data among clients for FL training and reserve some for global testing

### 2. **Federated Learning Setup**
- Use [Flower](https://flower.dev) framework for orchestrating federated training
- Clients are each trained on local data and communicate only weights with the server
- Server aggregates using **FedAvg strategy**

### 3. **Model Definition**
From `model.py`:  
A Keras regression model:
```text
Input → Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(1)
```
- Optimized using Adam
- Loss: Mean Squared Error (MSE)

### 4. **Clients (Local Training)**
From `client.py`:
- Each client gets initialized with a local dataset split
- Uses Flower’s `NumPyClient`:
  - `get_parameters`: Returns local model weights
  - `fit`: Trains model on local data for 5 epochs
  - `evaluate`: Evaluates local performance (loss, MAE)

### 5. **Server (Global Coordination & Evaluation)**
From `server.py`:
- Loads a **global test set**
- Evaluates using `mse`, `r²`
- Aggregates metrics from clients using weighted averaging
- Saves:
  - Final global model weights (`global_model_weights.pkl`)
  - Centralized training history (`history.pkl`)

### 6. **Main Script**
From `main.py`:
- Orchestrates multiple clients using threads
- Waits for all clients to complete
- Loads the saved global weights
- Evaluates final model on central test set
- Logs MSE and R²

### 7. **Visualization**
- `visualize_training.py`: Plots
  - Distributed vs centralized loss
  - Centralized MSE & R² over rounds
- `plot_predictions.py`: Plots actual vs predicted load from the final model

---

## 🔍 Quick Summary of Code Roles

| 📁 File | ⚙️ Purpose |
|--------|-----------|
| `model.py` | Defines the neural network model |
| `client.py` | Handles local training logic using Flower |
| `utils.py` | Preprocesses data, splits it for clients and testing |
| `server.py` | Coordinates training rounds, evaluates, saves weights/history |
| `main.py` | Runs client threads and evaluates the final model |
| `visualize_training.py` | Graphs training losses, MSE, R² over FL rounds |
| `plot_predictions.py` | Graphs actual vs predicted values from the final global model |

---