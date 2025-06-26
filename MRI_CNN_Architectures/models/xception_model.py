from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall


def build_model(input_shape=(224, 224, 3), num_classes=4):
    """Builds a classification model using Xception as the feature extractor."""
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="max",
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            Flatten(),
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
