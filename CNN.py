import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score, classification_report

from keras.models import load_model, Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import backend as K

import config
from read_files import load_arrays

K.set_image_data_format("channels_last")


def create_model(trial):
    model = Sequential()
    
    # First Conv1D layer
    model.add(Conv1D(
        filters=trial.suggest_int('conv1_filters', 32, 128),
        kernel_size=trial.suggest_int('conv1_kernel', 2, 5),
        activation=trial.suggest_categorical('conv1_activation', ['relu', 'elu', 'selu']),
        input_shape=(config.num_components, 1),
        padding="same"
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=trial.suggest_int('pool1_size', 2, 4)))
    
    # Second Conv1D layer
    model.add(Conv1D(
        filters=trial.suggest_int('conv2_filters', 64, 256),
        kernel_size=trial.suggest_int('conv2_kernel', 2, 5),
        activation=trial.suggest_categorical('conv2_activation', ['relu', 'elu', 'selu']),
        padding="same"
    ))
    model.add(BatchNormalization())
    
    # Third Conv1D layer
    model.add(Conv1D(
        filters=trial.suggest_int('conv3_filters', 128, 512),
        kernel_size=trial.suggest_int('conv3_kernel', 2, 5),
        activation=trial.suggest_categorical('conv3_activation', ['relu', 'elu', 'selu']),
        padding="same"
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=trial.suggest_int('pool2_size', 2, 4)))
    
    model.add(Flatten())
    
    # First Dense layer
    model.add(Dense(
        units=trial.suggest_int('dense1_units', 128, 512),
        activation=trial.suggest_categorical('dense1_activation', ['relu', 'elu', 'selu'])
    ))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout1', 0.1, 0.5)))
    
    # Second Dense layer
    model.add(Dense(
        units=trial.suggest_int('dense2_units', 64, 256),
        activation=trial.suggest_categorical('dense2_activation', ['relu', 'elu', 'selu'])
    ))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout2', 0.1, 0.5)))
    
    model.add(Dense(len(config.class_mapping), activation="softmax"))
    
    return model

def train_test_CNN(X_train, y_train, X_test, y_test, tune=False):
    # There are two options for training. If we're not tuning, a default model is created.
    # Else we can dig into hyperparameters to find the best model.

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    if not tune:
        print("Making Model")
        # Optuna is still used due to 'create_model' constraints, but not saved in the database
        model = create_model(optuna.trial.FixedTrial({
            'conv1_filters': 64,
            'conv1_kernel': 2,
            'conv1_activation': 'relu',
            'pool1_size': 2,
            'conv2_filters': 128,
            'conv2_kernel': 2,
            'conv2_activation': 'relu',
            'pool2_size': 2,
            'conv3_filters': 512,
            'conv3_kernel': 2,
            'conv3_activation': 'relu',
            'pool3_size': 2,
            'dense1_units': 256,
            'dense1_activation': 'relu',
            'dense2_units': 128,
            'dense2_activation': 'relu',
            'dropout1': 0.3,
            'dropout2': 0.1
        }))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model.summary()

        # reduce learning rate on plateau and early stopping
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001)
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            batch_size=4096,
            epochs=1,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr, early_stop],
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        # Confusion Matrix and Classification Report
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        cr = classification_report(y_true_classes, y_pred_classes, output_dict=True)

        # Convert classification report and history to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        history_serializable = convert_to_serializable(history.history)
        cr_serializable = convert_to_serializable(cr)
        cm_serializable = convert_to_serializable(cm)

        # Save metrics
        metrics = {
            'history': history_serializable,
            'test_accuracy': float(test_accuracy),  # Ensure these are converted as well
            'test_loss': float(test_loss),  # Ensure these are converted as well
            'confusion_matrix': cm_serializable,
            'classification_report': cr_serializable
        }

        with open('data/streamlit/model_metrics.json', 'w') as f:
            json.dump(metrics, f)

        # export model
        model.save("data/CNN_enmap_worldcover.h5")
        return model
    
    if tune:
        def objective(trial):
            # Create a model with hyperparameters suggested by Optuna
            model = create_model(trial)
            
            optimizer = Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=trial.suggest_uniform('lr_reduction_factor', 0.1, 0.5),
                patience=trial.suggest_int('lr_patience', 3, 10),
                min_lr=0.00001
            )
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=trial.suggest_int('early_stop_patience', 5, 15),
                restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                batch_size=trial.suggest_categorical('batch_size', [256, 1024, 2048, 4096, 8192]),
                epochs=trial.suggest_int('epochs', 10, 50),
                verbose=0,
                validation_data=(X_test, y_test),
                callbacks=[reduce_lr, early_stop],
            )
            
            return history.history['val_accuracy'][-1]

        # Create a study to store the results, or load an existing study
        study = optuna.create_study(
            study_name="cnn_hyperparameter_optimization",
            storage="sqlite:///optuna_study.db",  # SQLite database
            load_if_exists=True,
            direction='maximize'
        )
        study.optimize(objective, n_trials=40)

        # Take the best params for metrics
        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        # Train the model with the best hyperparameters
        best_model = create_model(optuna.trial.FixedTrial(best_params))
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        best_model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        best_model.fit(
            X_train,
            y_train,
            batch_size=best_params['batch_size'],
            epochs=50,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr, early_stop],
        )

        # export model
        best_model.save("data/CNN_enmap_worldcover_tuned.h5")
        return best_model


def predict_CNN():
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
            
            decomp = joblib.load('data/decomp_model.pkl')
            X_decomp = decomp.transform(X_filtered) 
            
            print("X_decomp shape: ", X_decomp.shape)
            break
    
    # Create a mask for valid labels (not -1 or 0)
    valid_mask = (y != -1) & (y != 0)

    # Convert y to consecutive labels, only for valid labels
    y_consecutive = np.full(y.shape, -1, dtype=int)  # Fill with -1 initially
    y_consecutive[valid_mask] = (y[valid_mask] / 10) - 1

    # Define a colormap and normalization for the predicted image
    cmap = plt.cm.get_cmap("tab10", len(config.unit_class_mapping))
    norm = mcolors.BoundaryNorm(
        boundaries=[key - 0.5 for key in sorted(config.unit_class_mapping.keys())]
        + [max(config.unit_class_mapping.keys()) + 0.5],
        ncolors=len(config.unit_class_mapping),
    )

    height, width = y.shape
    print(height, width)

    outputs = np.full((height, width), -1)  # Fill with -1 initially

    pixels = X_decomp
    positions = np.argwhere(valid_mask)

    model = load_model("data/CNN_enmap_worldcover.h5")

    if len(pixels) > 0:
        # Predict the labels for the valid pixels
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

        # Create a dictionary to store all the data
        streamlit_data = {
            'class_metrics': results.to_dict(orient='list'),
            'confusion_matrix': cm,
            'balanced_accuracy': balanced_acc,
            'predicted_outputs': outputs,
            'valid_mask': valid_mask,
            'correct_incorrect': mask_image
        }

        # Save all data to a single file
        joblib.dump(streamlit_data, 'data/streamlit/cnn_test_results.pkl')

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