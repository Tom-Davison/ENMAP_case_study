import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.colors as mcolors
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import config

def Patch(data, height_index, width_index, PATCH_SIZE):
    # transpose_array = data.transpose((2,0,1))
    # print transpose_array.shape
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch


def train_test_CNN(X_train, y_train, X_test, y_test):

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)


    print("Making Model")
    model = Sequential()
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=(30, 1),
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    #model.add(Conv1D(filters=256, kernel_size=3, activation="relu", padding="same"))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    #model.add(Dense(units=256, activation="relu"))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))
    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(len(config.class_mapping), activation="softmax"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        batch_size=2048,
        epochs=1,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop],
    )

    return model


def predict_CNN(model):
    # Create a mask for valid labels (not -1 or 0)
    valid_mask = (y != -1) & (y != 0)

    # Convert y to consecutive labels, only for valid labels
    y_consecutive = np.full(y.shape, -1, dtype=int)  # Fill with -1 initially
    y_consecutive[valid_mask] = (y[valid_mask] / 10) - 1

    cmap = plt.cm.get_cmap("tab10", len(unit_class_mapping))
    norm = mcolors.BoundaryNorm(
        boundaries=[key - 0.5 for key in sorted(unit_class_mapping.keys())]
        + [max(unit_class_mapping.keys()) + 0.5],
        ncolors=len(unit_class_mapping),
    )

    unique_labels = sorted(unit_class_mapping.keys())
    consecutive_to_label = {i: label for i, label in enumerate(unique_labels)}

    height, width = y.shape

    print(height, width)

    patches = []
    positions = []
    outputs = np.full((height, width), -1)  # Fill with -1 initially

    for i in range(height):
        for j in range(width):
            if valid_mask[i, j]:
                image_patch = Patch(X, i, j, PATCH_SIZE)
                patches.append(
                    image_patch.reshape(
                        1, image_patch.shape[2], image_patch.shape[0]
                    ).astype("float32")
                )
                positions.append((i, j))

    if patches:
        patches = np.concatenate(patches, axis=0)
        predictions = model.predict(patches)

        for prediction, position in zip(predictions, positions):
            outputs[position[0], position[1]] = np.argmax(prediction)  # + 1

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=1)
        # predicted_labels = predicted_labels -

        # Check unique labels and their counts
        uniqueLabels, labelCounts = np.unique(predicted_labels, return_counts=True)
        print("Unique Labels are: ", uniqueLabels)
        print("The number of predicted labels is: ", labelCounts)

        # Initialize dictionaries to count true positives and total samples per class
        class_counts = {
            key: {"true_positive": 0, "total": 0} for key in unit_class_mapping.keys()
        }

        # Iterate over the true labels and predictions
        for true_label, predicted_label in zip(
            y_consecutive[valid_mask].flatten(), outputs[valid_mask].flatten()
        ):
            true_label = int(true_label)
            predicted_label = int(predicted_label)

            class_counts[true_label]["total"] += 1
            if true_label == predicted_label:
                class_counts[true_label]["true_positive"] += 1

        # Calculate and print accuracy per class
        for class_id, counts in class_counts.items():
            if counts["total"] > 0:
                accuracy = counts["true_positive"] / counts["total"]
            else:
                accuracy = 0
            print(
                f"Accuracy for {class_mapping[unit_class_mapping[class_id]]}: {accuracy:.2f}"
            )

        # Create a mask for correct (green) and incorrect (red) labels
        correct_mask = (y_consecutive == outputs) & valid_mask
        incorrect_mask = (y_consecutive != outputs) & valid_mask

        # Create an RGB image to store the mask
        mask_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Color correct labels green (0, 255, 0)
        mask_image[correct_mask] = [0, 255, 0]

        # Color incorrect labels red (255, 0, 0)
        mask_image[incorrect_mask] = [255, 0, 0]

        # Plot the predicted image and the mask image side by side
        plt.figure(figsize=(20, 10))

        # Plot predicted image
        plt.subplot(1, 2, 1)
        predict_image = plt.imshow(outputs, cmap=cmap, norm=norm)
        cbar = plt.colorbar(predict_image, ticks=sorted(unit_class_mapping.keys()))
        cbar.ax.set_yticklabels(
            [
                class_mapping[unit_class_mapping[key]]
                for key in sorted(unit_class_mapping.keys())
            ]
        )
        plt.title("Predicted Image")

        # Plot correct vs. incorrect mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask_image)
        plt.title("Correct vs Incorrect Labels")

        plt.show()
        print("Plot done")
    else:
        print("No valid patches found for prediction.")
