from keras import utils as k_utils
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
import random
import scipy
import tqdm

import config
from read_files import load_arrays


def oversample_weak_classes(X, y):
    # Flatten y to 1D array for easier processing
    
    unique_labels, label_counts = np.unique(y, return_counts=True)
    max_count = np.max(label_counts)
    label_inverse_ratios = max_count / label_counts
    
    # Initialize new_X and new_Y with the oversampled data of the first label
    new_X = X[y == unique_labels[0], :].repeat(round(label_inverse_ratios[0]), axis=0)
    new_Y = y[y == unique_labels[0]].repeat(round(label_inverse_ratios[0]), axis=0)
    
    # Process the remaining labels
    for label, label_inverse_ratio in zip(unique_labels[1:], label_inverse_ratios[1:]):
        class_X = X[y == label, :].repeat(round(label_inverse_ratio), axis=0)
        class_Y = y[y == label].repeat(round(label_inverse_ratio), axis=0)
        new_X = np.concatenate((new_X, class_X))
        new_Y = np.concatenate((new_Y, class_Y))
    
    # Shuffle the new data
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(new_Y.shape[0])
    new_X = new_X[rand_perm, :]
    new_Y = new_Y[rand_perm]
    
    # Convert new_Y to integer type
    new_Y = new_Y.astype(int)
    
    # Print unique labels and their counts
    unique_labels, label_counts = np.unique(new_Y, return_counts=True)
    print("Unique Labels are: ", unique_labels)
    print("The number of augmented labels is: ", label_counts + 1)
    
    return new_X, new_Y


def apply_pca(X, y, num_components=30):
    X_reshaped = np.reshape(X, (-1, X.shape[2]))
    y_reshaped = y.flatten()
    
    valid_mask = (
        (~np.isnan(X_reshaped).any(axis=1)) & (y_reshaped != -1) & (y_reshaped != 0)
    )
    
    X_valid = X_reshaped[valid_mask]
    
    pca = PCA(n_components=num_components, whiten=True)
    X_pca = pca.fit_transform(X_valid)
    
    new_X = np.full((X_reshaped.shape[0], num_components), np.nan)
    
    new_X[valid_mask] = X_pca
    
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], num_components))
    
    return new_X


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


def AugmentData(X_train):
    for i in range(int(X_train.shape[0] / 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if num == 0:
            flipped_patch = np.flipud(patch)
        if num == 1:
            flipped_patch = np.fliplr(patch)
        if num == 2:
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.rotate(
                patch,
                no,
                axes=(1, 0),
                reshape=False,
                output=None,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
    patch2 = flipped_patch
    X_train[i, :, :, :] = patch2
    return X_train


def process_data():

    X_list = []
    y_list = []
    for paths in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data with PCA"):
        if paths["usage"] == "training":
            X, y = load_arrays(paths["area_code"])
        
            y = y.flatten()
            
            valid_mask = (y != -1) & (y != 0)
            X_filtered = X.reshape(-1, X.shape[-1])[valid_mask]
            y_filtered = y[valid_mask]
            
            pca = PCA(n_components=config.num_components, whiten=True)
            X_pca = pca.fit_transform(X_filtered)
            
            X_list.append(X_pca)
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
    X_conc_samp, y_conc_samp = oversample_weak_classes(X_conc, y_conc)
    #save training library here
    print("Splitting Train Test Set")
    X_train, X_test, y_train, y_test = train_test_split(X_conc_samp, y_conc_samp, test_size=0.2, stratify=y_conc_samp)

    y_train_norm = (y_train // 10) - 1
    y_test_norm = (y_test // 10) - 1

    y_train = k_utils.to_categorical(y_train_norm)
    y_test = k_utils.to_categorical(y_test_norm)

    return X_train, X_test, y_train, y_test

