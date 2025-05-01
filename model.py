from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.losses import Huber
import tensorflow.keras.layers as layers
import tensorflow.keras.ops as ops

import tensorflow as tf


# Custom Yogi Optimizer
class Yogi(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, name="Yogi", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialize dictionary to store slots using variable references
        self._slots = {}

    def update_step(self, gradient, variable, learning_rate=None):
        # Get hyperparameters
        lr = self.learning_rate if learning_rate is None else learning_rate
        beta1 = self.beta1
        beta2 = self.beta2

        # Use variable.ref() as the key (hashable reference to the variable)
        var_ref = variable.ref()

        # Initialize or get slots for this variable
        if var_ref not in self._slots:
            self._slots[var_ref] = {
                "m": tf.Variable(tf.zeros_like(variable), trainable=False),
                "v": tf.Variable(tf.zeros_like(variable), trainable=False)
            }
        m = self._slots[var_ref]["m"]
        v = self._slots[var_ref]["v"]

        # Yogi update rules
        m_t = beta1 * m + (1 - beta1) * gradient  # Momentum update (same as Adam)
        v_t = v - (1 - beta2) * tf.sign(v - tf.square(gradient)) * tf.square(gradient)  # Yogi variance update
        variable.assign(variable - lr * m_t / (tf.sqrt(v_t) + self.epsilon))  # Weight update

        # Update slots
        m.assign(m_t)
        v.assign(v_t)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Optional: Keep this for low-level control if needed, but update_step should handle the main logic
        self.update_step(grad, var)
        return tf.no_op()

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Sparse gradient support (optional, can be simplified for now)
        raise NotImplementedError("Sparse gradient updates not implemented for Yogi yet.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate.numpy() if hasattr(self.learning_rate, "numpy") else self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        })
        return config
    
# Custom Layer for Reducing Mean Over Time Axis
class ReduceMeanLayer(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return ops.mean(inputs, axis=self.axis)

# Model Creation Function with STT, LSTM, and CNN
def create_model(input_shape, optimizer_type="yogi"):
    input_layer = Input(shape=(input_shape, 1))  # LSTM/CNN/STT expects 3D shape: (timesteps, features, channels)

    # STT Path: Short-Term Transformer for sequential patterns
    stt_output = MultiHeadAttention(key_dim=1, num_heads=2, value_dim=1)(input_layer, input_layer)
    stt_output = LayerNormalization()(stt_output)  # Normalize the output
    stt_output = ReduceMeanLayer(axis=1)(stt_output)  # Aggregate over time steps to match other paths

    # CNN Path: Extract spatial features
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)

    # LSTM Path: Capture temporal trends
    lstm = LSTM(64, return_sequences=False)(input_layer)

    # Merge STT, CNN, and LSTM outputs
    merged = concatenate([stt_output, cnn, lstm])
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(0.3)(dense)
    dense = Dense(32, activation='relu')(dense)
    output = Dense(1)(dense)  # Regression output

    model = Model(inputs=input_layer, outputs=output)

    # Select optimizer based on input
    if optimizer_type.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    elif optimizer_type.lower() == "yogi":
        optimizer = Yogi(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'adam' or 'yogi'.")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # model.compile(optimizer=optimizer, loss=Huber(delta=1.0), metrics=['mae'])

    return model

# Example usage (optional, for testing)
if __name__ == "__main__":
    model = create_model(input_shape=10, optimizer_type="yogi")
    model.summary()