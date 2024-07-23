import numpy as np
from keras import backend as K

from read_files import read_files
from process_files import process_data
from CNN import train_test_CNN, predict_CNN

# import tensorflow_probability as tfp
# import warnings
# warnings.filterwarnings("ignore")
K.set_image_data_format("channels_last")

# Paths to your data
enmap_data_path = "data/ENMAP01-____L2A-DT0000001049_20220612T105735Z_028_V010303_20230922T131826Z-SPECTRAL_IMAGE.TIF"
enmap_metadata_path = "data/ENMAP01-____L2A-DT0000001049_20220612T105735Z_028_V010303_20230922T131826Z-METADATA.XML"
esa_worldcover_path = "data/ESA_WorldCover_10m_2021_v200_N45E009_Map.TIF"

class_mapping = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built up",
    60: "bare/sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    100: "moss and lichen",
}

# class_index_mapping = {i: key for i, key in enumerate(class_mapping.keys())}
# class_reverse_index_mapping = {key: i for i, key in enumerate(class_mapping.keys())}
unit_class_mapping = {
    new_index: original_key
    for new_index, original_key in enumerate(class_mapping.keys())
}

# define params
numComponents = 30
windowSize = 1
testRatio = 0.25
PATCH_SIZE = 1

# read files
X, y = read_files(enmap_data_path, enmap_metadata_path, esa_worldcover_path, plot=False)

unique_labels = np.unique(y)
n_classes = len(unique_labels)

y_filtered = y[(y != -1) & (y != 0)]
unique_labels_filtered = np.unique(y_filtered)
n_classes_filtered = len(unique_labels_filtered)

uniqueLabels, labelCounts = np.unique(y, return_counts=True)
print("Unique Labels are: ", uniqueLabels)
print("The number of labels is: ", labelCounts + 1)

missing_classes = set(class_mapping.keys()) - set(unique_labels)
if missing_classes:
    raise ValueError(
        f"CRITICAL: Some classes are missing in the data! Missing classes: {missing_classes}"
    )

# processing
X_train, X_test, y_train, y_test, X = process_data(
    X,
    y,
    numComponents=numComponents,
    windowSize=windowSize,
    testRatio=testRatio,
    PATCH_SIZE=PATCH_SIZE,
)

# modelling
model = train_test_CNN(X_train, y_train, X_test, y_test, n_classes_filtered)
predict_CNN(model, X, y, unit_class_mapping, class_mapping, PATCH_SIZE)


# testing on novel data
enmap_data_path = "data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-SPECTRAL_IMAGE.tif"
enmap_metadata_path = "data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-METADATA.xml"
esa_worldcover_path = "data/ESA_WorldCover_10m_2021_v200_N51E006_Map.tif"

X, y = read_files(enmap_data_path, enmap_metadata_path, esa_worldcover_path, plot=False)
unique_labels = np.unique(y)
n_classes = len(unique_labels)
uniqueLabels, labelCounts = np.unique(y, return_counts=True)
print("Unique Labels are: ", uniqueLabels)
print("The number of labels is: ", labelCounts + 1)

# processing
X_train, X_test, y_train, y_test, X = process_data(
    X,
    y,
    numComponents=numComponents,
    windowSize=windowSize,
    testRatio=testRatio,
    PATCH_SIZE=PATCH_SIZE,
)
predict_CNN(model, X, y, unit_class_mapping, class_mapping, PATCH_SIZE)
