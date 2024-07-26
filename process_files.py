from keras import utils as k_utils
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random
import scipy
import tqdm
import joblib
from collections import defaultdict

import config
from read_files import load_arrays


def oversample_and_cap_classes(X, y, class_cap):
    unique_labels, label_counts = np.unique(y, return_counts=True)
    
    new_X = []
    new_Y = []
    
    for label, count in zip(unique_labels, label_counts):
        class_X = X[y == label]
        class_Y = y[y == label]
        
        if count > class_cap:
            # Randomly sample class_cap samples
            indices = np.random.choice(count, class_cap, replace=False)
            class_X = class_X[indices]
            class_Y = class_Y[indices]
        elif count < class_cap:
            # Oversample to reach class_cap
            repeats = class_cap // count
            remainder = class_cap % count
            class_X = np.repeat(class_X, repeats, axis=0)
            class_Y = np.repeat(class_Y, repeats)
            
            if remainder > 0:
                indices = np.random.choice(count, remainder, replace=False)
                class_X = np.vstack((class_X, class_X[indices]))
                class_Y = np.concatenate((class_Y, class_Y[indices]))
        
        new_X.append(class_X)
        new_Y.append(class_Y)
        
        # Print the ratio of values in the class vs class_cap
        ratio = count / class_cap
        print(f"Class {label}: Original count = {count}, Ratio to class_cap = {ratio:.2f}")
    
    new_X = np.vstack(new_X)
    new_Y = np.concatenate(new_Y)
    
    # Shuffle the new data
    print("Shuffling data")
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(new_Y.shape[0])
    new_X = new_X[rand_perm]
    new_Y = new_Y[rand_perm]
    
    # Convert new_Y to integer type
    new_Y = new_Y.astype(int)
    
    # Print unique labels and their counts
    unique_labels, label_counts = np.unique(new_Y, return_counts=True)
    print("\nAfter balancing:")
    print("Unique Labels are:", unique_labels)
    print("The number of samples per class is:", label_counts)
    
    return new_X, new_Y


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX


def createPatches(X, y, windowSize=1, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])
    )
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[
                r - margin : r + margin + 1, c - margin : c + margin + 1
            ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels = patchesLabels / 10
        patchesLabels -= 1
    return patchesData, patchesLabels


def prep_training_data(regenerate_library=False):

    if regenerate_library:
        X_list = []
        y_list = []
        ica = joblib.load('data/decomp_model.pkl')
        for paths in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data with Decomposition"):
            if paths["usage"] == "training":
                X, y = load_arrays(paths["area_code"])
            
                y = y.flatten()
                
                valid_mask = (y != -1) & (y != 0)
                X_filtered = X.reshape(-1, X.shape[-1])[valid_mask]
                y_filtered = y[valid_mask]
                
                X_decomp = ica.transform(X_filtered)
                
                X_list.append(X_decomp)
                y_list.append(y_filtered.reshape(-1, 1))
        
        X_conc = np.concatenate(X_list)
        y_conc = np.concatenate(y_list)
        y_conc = y_conc.flatten()

        print("Done loading data")
        unique_labels = np.unique(y_conc)
        print("Unique Labels are: ", unique_labels)
        missing_classes = set(config.class_mapping.keys()) - set(unique_labels)

        if missing_classes:
            raise ValueError(
                f"CRITICAL: Some classes are missing in the data! Missing classes: {missing_classes}"
            )

        print("Oversampling Weak Classes")
        X_conc_samp, y_conc_samp = oversample_and_cap_classes(X_conc, y_conc, config.sample_cap)

        print("Saving training library:")
        np.save("data/X_train.npy", X_conc_samp)
        np.save("data/y_train.npy", y_conc_samp)
    
    else:
        print("Loading training library:")
        X_conc_samp = np.load("data/X_train.npy")
        y_conc_samp = np.load("data/y_train.npy")

    print("Splitting Train Test Set")
    X_train, X_test, y_train, y_test = train_test_split(X_conc_samp, y_conc_samp, test_size=0.2, stratify=y_conc_samp)

    y_train_norm = (y_train // 10) - 1
    y_test_norm = (y_test // 10) - 1

    y_train = k_utils.to_categorical(y_train_norm)
    y_test = k_utils.to_categorical(y_test_norm)

    return X_train, X_test, y_train, y_test

def build_balanced_sample(target_samples_per_class=10000, max_samples_per_image=100000):
    all_samples = defaultdict(list)

    for path in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data for Decomposition"):
        if path["usage"] == "training":
            X, y = load_arrays(path["area_code"])
            
            y = y.flatten()
            valid_mask = (y != -1) & (y != 0)
            X_filtered = X.reshape(-1, X.shape[-1])[valid_mask]
            y_filtered = y[valid_mask]
            
            if len(X_filtered) > max_samples_per_image:
                indices = np.random.choice(len(X_filtered), max_samples_per_image, replace=False)
                X_filtered = X_filtered[indices]
                y_filtered = y_filtered[indices]
            
            for class_label in np.unique(y_filtered):
                class_indices = np.where(y_filtered == class_label)[0]
                all_samples[class_label].extend(X_filtered[class_indices].tolist())
        
    # Balance the classes
    balanced_samples = []
    balanced_labels = []
    
    print('Balance')
    for class_label, samples in all_samples.items():
        if len(samples) > target_samples_per_class:
            selected_indices = np.random.choice(len(samples), target_samples_per_class, replace=False)
            balanced_samples.extend([samples[i] for i in selected_indices])
            balanced_labels.extend([class_label] * target_samples_per_class)
        else:
            # If we don't have enough samples, use all of them
            balanced_samples.extend(samples)
            balanced_labels.extend([class_label] * len(samples))
    
    return np.array(balanced_samples), np.array(balanced_labels)



def generate_decomposition(model_path='data/decomp_model.pkl'):

    X_balanced, y_balanced = build_balanced_sample()
    
    # Create and fit FastICA model
    print('Decompose')
    ica = FastICA(n_components=config.num_components, random_state=42, max_iter=1000)
    ica.fit(X_balanced)
    
    # Export the model
    joblib.dump(ica, model_path)
    
    print(f"FastICA model created and saved to {model_path}")
    