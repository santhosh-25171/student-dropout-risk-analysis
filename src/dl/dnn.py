"""
Deep Neural Network (DNN) Architecture with Batch Normalization and Dropout Regularization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dnn(input_dim: int, learning_rate: float = 0.001, dropout_rate: float = 0.3) -> keras.Model:
    """
    Build a Deep Neural Network with Batch Normalization and Dropout regularization.
    Designed to prevent overfitting while learning hierarchical representations.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        
        layers.Dense(128, use_bias=False, name="dense_block_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation("relu", name="relu_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),
        
        layers.Dense(64, use_bias=False, name="dense_block_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation("relu", name="relu_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),
        
        layers.Dense(32, use_bias=False, name="dense_block_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.Activation("relu", name="relu_3"),
        layers.Dropout(dropout_rate * 0.7, name="dropout_3"),
        
        layers.Dense(16, activation="relu", name="dense_block_4"),
        
        layers.Dense(1, activation="sigmoid", name="dropout_risk_output")
    ], name="EDUVISION_DNN")

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
