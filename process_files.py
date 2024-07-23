from keras import utils as k_utils
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
import random
import scipy


def splitTrainTestSet(X, y, testRatio=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=345, stratify=y
    )
    return X_train, X_test, y_train, y_test


def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    newX = X[y == uniqueLabels[0], :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :]
    newY = newY[rand_perm]
    newY = newY.astype(int)
    uniqueLabels, labelCounts = np.unique(newY, return_counts=True)
    print("Unique Labels are: ", uniqueLabels)
    print("The number of augmented labels is: ", labelCounts + 1)
    return newX, newY


def applyPCA(X, y, numComponents=75):
    X_reshaped = np.reshape(X, (-1, X.shape[2]))
    y_reshaped = y.flatten()
    valid_mask = (
        (~np.isnan(X_reshaped).any(axis=1)) & (y_reshaped != -1) & (y_reshaped != 0)
    )
    X_valid = X_reshaped[valid_mask]
    pca = PCA(n_components=numComponents, whiten=True)
    X_pca = pca.fit_transform(X_valid)
    newX = np.full((X_reshaped.shape[0], numComponents), np.nan)
    newX[valid_mask] = X_pca
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX, pca


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


def process_data(X, y, numComponents=30, windowSize=1, testRatio=0.25, PATCH_SIZE=1):

    print("Doing PCA")
    X, pca = applyPCA(X, y, numComponents=numComponents)
    print("Extracting Patches")
    XPatches, yPatches = createPatches(X, y, windowSize=1)
    print("Oversampling Weak Classes")
    XPatches, yPatches = oversampleWeakClasses(XPatches, yPatches)
    print(XPatches.shape, yPatches.shape)
    print("Splitting Train Test Set")
    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
    print("Augmenting Data")
    # X_train = AugmentData(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[3])
    y_train = k_utils.to_categorical(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    return X_train, X_test, y_train, y_test, X
