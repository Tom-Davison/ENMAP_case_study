import matplotlib.pyplot as plt
import tqdm
import joblib
from sklearn.decomposition import PCA
from keras.models import load_model
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
import pandas as pd
from keras import backend as K

import config
from read_files import load_arrays

K.set_image_data_format("channels_last")


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
    #model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(config.num_components, 1), padding="same"))
    #model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu",  input_shape=(config.num_components, 1), padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=512, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
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
        batch_size=4096,
        epochs=20,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop],
    )

    #export model
    model.save("data/CNN_enmap_worldcover.h5")
    return model


def predict_CNN(model):
    for paths in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data with PCA"):
        if paths["usage"] == "testing":
            X, y = load_arrays(paths["area_code"])
            print("X shape: ", X.shape)
            print("y shape: ", y.shape)
            
            # Create the valid mask
            valid_mask = (y != -1) & (y != 0)
            
            # Reshape X and apply the mask
            X_reshaped = X.reshape(-1, X.shape[-1])
            valid_mask_flattened = valid_mask.flatten()
            X_filtered = X_reshaped[valid_mask_flattened]
            y_filtered = y[valid_mask]

            print("X_filtered shape: ", X_filtered.shape)
            print("y_filtered shape: ", y_filtered.shape)
            
            print("BACKUP!!!")
            pca = joblib.load('data/decomp_model_backup.pkl')
            X_decomp = pca.transform(X_filtered) 
            
            print("X_decomp shape: ", X_decomp.shape)
            break
    
    # Create a mask for valid labels (not -1 or 0)
    valid_mask = (y != -1) & (y != 0)

    # Convert y to consecutive labels, only for valid labels
    y_consecutive = np.full(y.shape, -1, dtype=int)  # Fill with -1 initially
    y_consecutive[valid_mask] = (y[valid_mask] / 10) - 1

    cmap = plt.cm.get_cmap("tab10", len(config.unit_class_mapping))
    norm = mcolors.BoundaryNorm(
        boundaries=[key - 0.5 for key in sorted(config.unit_class_mapping.keys())]
        + [max(config.unit_class_mapping.keys()) + 0.5],
        ncolors=len(config.unit_class_mapping),
    )

    unique_labels = sorted(config.unit_class_mapping.keys())

    height, width = y.shape
    print(height, width)

    outputs = np.full((height, width), -1)  # Fill with -1 initially

    # Use X_pca directly as pixels and get positions of valid pixels
    pixels = X_decomp
    positions = np.argwhere(valid_mask)

    model = load_model("data/CNN_enmap_worldcover_backup.h5")

    if len(pixels) > 0:
        predictions = model.predict(pixels)

        for prediction, position in zip(predictions, positions):
            outputs[position[0], position[1]] = np.argmax(prediction)

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = y_consecutive[valid_mask].flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Calculate precision, recall, and F1-score for each class
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels)

        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        # Create a DataFrame to display results
        results = pd.DataFrame({
            'Class': [config.class_mapping[config.unit_class_mapping[i]] for i in range(len(precision))],
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Support': support
        })

        print("Performance Metrics per Class:")
        print(results)
        print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

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
        cbar = plt.colorbar(predict_image, ticks=sorted(config.unit_class_mapping.keys()))
        cbar.ax.set_yticklabels(
            [
                config.class_mapping[config.unit_class_mapping[key]]
                for key in sorted(config.unit_class_mapping.keys())
            ]
        )
        plt.title("Predicted Image")

        # Plot correct vs. incorrect mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask_image)
        plt.title("Correct vs Incorrect Labels")

        plt.show()
        print("Plot done")