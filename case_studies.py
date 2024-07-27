import matplotlib.pyplot as plt
import tqdm
import joblib
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
import pandas as pd
from keras.models import load_model
import xml.etree.ElementTree as ET

import config
from read_files import load_arrays


def generate_case_1():
    for paths in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data with PCA"):
        if paths["usage"] == "case_study_1":
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
            
            pca = joblib.load('data/decomp_model.pkl')
            X_decomp = pca.transform(X_filtered) 
            
            print("X_decomp shape: ", X_decomp.shape)
    
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

            model = load_model("data/CNN_enmap_worldcover.h5")

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

def generate_case_2():
    results_df = pd.DataFrame(columns=['Date'] + list(config.unit_class_mapping.values()))
    for paths in config.enmap_data.values():
        if paths["usage"] == "case_study_2":

            # Read XML for boundary
            tree = ET.parse(paths["metadata"])
            root = tree.getroot()

            # Extract temporal coverage start and stop times
            temporal_coverage = root.find(".//temporalCoverage")
            start_time = temporal_coverage.find("startTime").text

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

            pca = joblib.load('data/decomp_model.pkl')
            X_decomp = pca.transform(X_filtered) 
            
            print("X_decomp shape: ", X_decomp.shape)
    
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

            model = load_model("data/CNN_enmap_worldcover.h5")

            if len(pixels) > 0:
                predictions = model.predict(pixels)

                for prediction, position in zip(predictions, positions):
                    outputs[position[0], position[1]] = np.argmax(prediction)

                # Calculate the fraction of each land-type
                unique, counts = np.unique(outputs, return_counts=True)
                total_counts = np.sum(counts)
                fractions = {config.unit_class_mapping.get(label, 'Unknown'): count / total_counts 
                             for label, count in zip(unique, counts)}
                
                # Ensure all land types are included, even if count is zero
                for land_type in config.unit_class_mapping.values():
                    if land_type not in fractions:
                        fractions[land_type] = 0.0
                
                # Add the date to the fractions dictionary
                fractions['Date'] = start_time
                
                # Create a DataFrame from the fractions dictionary
                fractions_df = pd.DataFrame([fractions])
                
                # Append the fractions DataFrame to the results DataFrame
                results_df = pd.concat([results_df, fractions_df], ignore_index=True)

                # Plot the predicted image and the mask image side by side
                plt.figure(figsize=(10, 10))

                predict_image = plt.imshow(outputs, cmap=cmap, norm=norm)
                cbar = plt.colorbar(predict_image, ticks=sorted(config.unit_class_mapping.keys()))
                cbar.ax.set_yticklabels(
                    [
                        config.class_mapping[config.unit_class_mapping[key]]
                        for key in sorted(config.unit_class_mapping.keys())
                    ]
                )
                plt.title("Predicted Image")

                plt.show()

    # Print the results dataframe
    print(results_df)