import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


def train_dnn(X, y, epochs=50, batch_size=16):

    le = LabelEncoder()

    y_enc = le.fit_transform(y)

    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.2,
        random_state=42
    )

    model = Sequential([

        Dense(256, activation='relu', input_shape=(X.shape[1],)),

        Dropout(0.3),

        Dense(128, activation='relu'),

        Dropout(0.3),

        Dense(y_cat.shape[1], activation='softmax')

    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Accuracy: {acc*100:.2f}%")

    # Predictions
    y_probs = model.predict(X_test)

    y_pred = np.argmax(y_probs, axis=1)

    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")

    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:\n")

    print(confusion_matrix(y_true, y_pred))

    # Accuracy Graph
    plt.figure(figsize=(10,6))

    plt.plot(history.history['accuracy'], label='Training Accuracy')

    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Accuracy vs Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.grid(True)

    plt.savefig("assets/accuracy_graph.png")

    plt.show()

    # Loss Graph
    plt.figure(figsize=(10,6))

    plt.plot(history.history['loss'], label='Training Loss')

    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Loss vs Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)

    plt.savefig("assets/loss_graph.png")

    plt.show()

    return model, le