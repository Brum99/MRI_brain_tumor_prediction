from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dropout,
    Dense,
)
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall


def _base_cnn(input_shape):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Flatten(),
        ]
    )
    return model


def build_model(input_shape=(224, 224, 3), num_classes=4):
    """Builds a custom CNN classifier using the shared top layers."""
    base_model = _base_cnn(input_shape)

    model = Sequential(
        [
            base_model,
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.25),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()],
    )
    return model
