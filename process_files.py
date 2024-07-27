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
    # Oversample and cap the classes to ensure balanced consistency when training.
    # This ensures that training does not bias towards the most common classes

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
        
        # Print the ratio of values in the class vs class_cap for debug
        ratio = count / class_cap
        print(f"Class {label}: Original count = {count}, Ratio to class_cap = {ratio:.2f}")
    
    new_X = np.vstack(new_X)
    new_Y = np.concatenate(new_Y)
    
    # Shuffle the new data just in case train/test split is not random
    print("Shuffling data")
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(new_Y.shape[0])
    new_X = new_X[rand_perm]
    new_Y = new_Y[rand_perm]
    new_Y = new_Y.astype(int)
    
    # Print unique labels and their counts for debug
    unique_labels, label_counts = np.unique(new_Y, return_counts=True)
    print("\nAfter balancing:")
    print("Unique Labels are:", unique_labels)
    print("The number of samples per class is:", label_counts)
    
    return new_X, new_Y


def prep_training_data(regenerate_library=False):

    # Load the data and apply decomposition (unless it has already been done)
    if regenerate_library:
        X_list = []
        y_list = []
        ica = joblib.load('data/decomp_model.pkl') # Load the decomposition model
        for paths in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data with Decomposition"):
            if paths["usage"] == "training":
                # Load the data reshape
                X, y = load_arrays(paths["area_code"])
            
                y = y.flatten()
                
                valid_mask = (y != -1) & (y != 0)
                X_filtered = X.reshape(-1, X.shape[-1])[valid_mask]
                y_filtered = y[valid_mask]
                
                X_decomp = ica.transform(X_filtered) # Apply the decomposition model
                
                X_list.append(X_decomp)
                y_list.append(y_filtered.reshape(-1, 1))
        
        X_conc = np.concatenate(X_list)
        y_conc = np.concatenate(y_list)
        y_conc = y_conc.flatten()

        print("Done loading data")
        unique_labels = np.unique(y_conc)
        print("Unique Labels are: ", unique_labels)
        missing_classes = set(config.class_mapping.keys()) - set(unique_labels)

        # Check if any classes are missing. This will cause errors later on if the sampling has failed
        if missing_classes:
            raise ValueError(
                f"CRITICAL: Some classes are missing in the data! Missing classes: {missing_classes}"
            )

        # Oversample and cap the classes to ensure balanced consistency in training
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

    # Convert the labels to categorical; the initial start at 10 confuses one-hot. TODO: I think there's a non-0-indexed method. Check this later
    y_train_norm = (y_train // 10) - 1
    y_test_norm = (y_test // 10) - 1

    y_train = k_utils.to_categorical(y_train_norm)
    y_test = k_utils.to_categorical(y_test_norm)

    return X_train, X_test, y_train, y_test

def build_balanced_sample(target_samples_per_class=10000, max_samples_per_image=100000):
    # This function aims to balance target samples to allow for better classification
    # As this data is only used for decomposition, we can afford to have a smaller sample size
    # This ensures that the decomposition model is not biased towards the most common classes

    all_samples = defaultdict(list)

    # Iterate over all the training data
    for path in tqdm.tqdm(config.enmap_data.values(), desc="Loading Data for Decomposition"):
        if path["usage"] == "training":
            X, y = load_arrays(path["area_code"])
            
            y = y.flatten()
            valid_mask = (y != -1) & (y != 0)
            X_filtered = X.reshape(-1, X.shape[-1])[valid_mask]
            y_filtered = y[valid_mask]
            
            # Limit the number of samples per image (memory constraint)
            if len(X_filtered) > max_samples_per_image:
                indices = np.random.choice(len(X_filtered), max_samples_per_image, replace=False)
                X_filtered = X_filtered[indices]
                y_filtered = y_filtered[indices]
            
            # Group the samples by class
            for class_label in np.unique(y_filtered):
                class_indices = np.where(y_filtered == class_label)[0]
                all_samples[class_label].extend(X_filtered[class_indices].tolist())
        
    # Balance the classes
    balanced_samples = []
    balanced_labels = []
    
    print('Balancing classes')
    for class_label, samples in all_samples.items():
        # Take a random sample of the target size
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
    
    # Create and fit FastICA model using the balanced data.
    # ICA is chosen over PCA due to the nature of hyperspectral data. 
    # Hyperspectral can have mixed pixel spectra due to the presence of multiple materials 
    # within a single pixel. ICA works to unmix these spectra into their components, making 
    # it easier to identify and classify the underlying materials. 
    print('Decompose')
    ica = FastICA(n_components=config.num_components, random_state=42, max_iter=1000)
    ica.fit(X_balanced)
    
    # Export the model for later use
    joblib.dump(ica, model_path)
    
    print(f"FastICA model created and saved to {model_path}")
    