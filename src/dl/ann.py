"""
Artificial Neural Network (ANN) Architecture using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_ann(input_dim: int, learning_rate: float = 0.001) -> keras.Model:
    """
    Build a standard Artificial Neural Network with two dense hidden layers.
    Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(64, activation="relu", name="dense_hidden_1"),
        layers.Dense(32, activation="relu", name="dense_hidden_2"),
        layers.Dense(1, activation="sigmoid", name="dropout_risk_output")
    ], name="EDUVISION_ANN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model
