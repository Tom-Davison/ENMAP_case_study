import xml.etree.ElementTree as ET
import rasterio
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna

# Paths to your data
enmap_data_path = 'data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-SPECTRAL_IMAGE.tif'
enmap_metadata_path = 'data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-METADATA.xml'
esa_worldcover_path = 'data/ESA_WorldCover_10m_2021_v200_N51E006_Map.tif'

# Load and parse the XML metadata file
tree = ET.parse(enmap_metadata_path)
root = tree.getroot()

# Extract bounding coordinates
bounding_polygon = root.find(".//spatialCoverage/boundingPolygon")
coordinates = []
for point in bounding_polygon.findall("point"):
    lat = float(point.find("latitude").text)
    lon = float(point.find("longitude").text)
    coordinates.append((lat, lon))

print("Bounding Coordinates:", coordinates)

# Load the hyperspectral data
with rasterio.open(enmap_data_path) as src:
    hyperspectral_data = src.read()
    transform = src.transform
    crs = src.crs

# Confirm the CRS
print(f"CRS of the hyperspectral data: {crs}")

# Get coordinates of each pixel in projected coordinates
n_bands, n_rows, n_cols = hyperspectral_data.shape

# Using the affine transform directly to get coordinates of each pixel
projected_coords = np.array([[transform * (col, row) for col in range(n_cols)] for row in range(n_rows)])

# Transform projected coordinates to geographic coordinates
transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
geographic_coords = np.array([[transformer.transform(x, y) for x, y in row] for row in projected_coords])

# Load the ESA WorldCover data
with rasterio.open(esa_worldcover_path) as wc_src:
    worldcover_data = wc_src.read(1)
    wc_transform = wc_src.transform
    wc_crs = wc_src.crs

# Confirm the CRS
print(f"CRS of the WorldCover data: {wc_crs}")

# Transform geographic coordinates of EnMAP to ESA WorldCover projection if necessary
if wc_crs != 'EPSG:4326':
    transformer = Transformer.from_crs('EPSG:4326', wc_crs, always_xy=True)
    coords_wc = np.array([[transformer.transform(lon, lat) for lon, lat in row] for row in geographic_coords])
else:
    coords_wc = geographic_coords

def get_worldcover_label(lat, lon):
    # Convert lat/lon to row/col in WorldCover raster
    row, col = ~wc_transform * (lon, lat)
    row, col = int(row), int(col)
    if 0 <= row < worldcover_data.shape[0] and 0 <= col < worldcover_data.shape[1]:
        return worldcover_data[row, col]
    else:
        return None

# Get land cover labels for all coordinates
land_cover_labels = np.array([[get_worldcover_label(lat, lon) for lon, lat in row] for row in coords_wc])
print("Land Cover Labels:", land_cover_labels)

# Debugging: Check if all labels are None
if np.all(land_cover_labels == None):
    print("All labels are None. Check coordinate transformation and data alignment.")
else:
    print("Some labels are successfully extracted.")

# Calculate extents for both datasets
wc_extent = [
    wc_transform[2], wc_transform[2] + wc_transform[0] * worldcover_data.shape[1],
    wc_transform[5] + wc_transform[4] * worldcover_data.shape[0], wc_transform[5]
]

enmap_extent = [
    np.min(geographic_coords[:, :, 0]), np.max(geographic_coords[:, :, 0]),
    np.min(geographic_coords[:, :, 1]), np.max(geographic_coords[:, :, 1])
]

print("WorldCover Extent:", wc_extent)
print("EnMAP Extent:", enmap_extent)

# Visualize the overlay of EnMAP on ESA WorldCover
fig, ax = plt.subplots(figsize=(10, 10))

# Plot ESA WorldCover data
img_wc = ax.imshow(worldcover_data, cmap='tab20', extent=wc_extent)

# Overlay EnMAP first band data
img_enmap = ax.imshow(hyperspectral_data[0, :, :], cmap='gray', alpha=0.75, extent=enmap_extent, vmin=np.nanmin(hyperspectral_data[0, :, :]), vmax=np.nanmax(hyperspectral_data[0, :, :]))

ax.set_title('EnMAP Overlay on ESA WorldCover')
plt.colorbar(img_wc, ax=ax, orientation='vertical', label='ESA WorldCover Classes')
plt.colorbar(img_enmap, ax=ax, orientation='horizontal', label='EnMAP First Band', pad=0.1)
plt.show()






# Prepare the data
# Assuming hyperspectral_data and land_cover_labels are already defined
n_bands, n_rows, n_cols = hyperspectral_data.shape
X = np.moveaxis(hyperspectral_data, 0, -1)  # Shape: (n_rows, n_cols, n_bands)
y = land_cover_labels

# Remove pixels with None labels
mask = y != None
X = X[mask]
y = y[mask]

# Flatten the 2D spatial dimensions to 1D for CNN input
X = X.reshape(-1, n_bands)
y = y.flatten()

# Check unique labels and determine the number of classes
unique_labels = np.unique(y)
n_classes = len(unique_labels)
print(f"Unique labels: {unique_labels}")
print(f"Number of classes: {n_classes}")

# Ensure all labels are valid
y = np.where(np.isin(y, unique_labels), y, 0)  # Replace invalid labels with a default value (e.g., 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for the CNN
X_train = X_train.reshape(-1, 1, 1, n_bands)
X_test = X_test.reshape(-1, 1, 1, n_bands)

# Encode the labels using unique labels
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 4)
    conv_layers = [trial.suggest_int(f'n_filters_{i}', 32, 256) for i in range(n_conv_layers)]
    kernel_sizes = [trial.suggest_int(f'kernel_size_{i}', 1, 3) for i in range(n_conv_layers)]
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
    dense_layers = [trial.suggest_int(f'n_units_{i}', 64, 512) for i in range(n_dense_layers)]
    pooling_type = trial.suggest_categorical('pooling_type', ['max', 'average'])
    pooling_size = trial.suggest_int('pooling_size', 1, 3)  # Allow larger pooling sizes
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Build the model with suggested hyperparameters
    model = build_model((1, 1, n_bands), n_classes, conv_layers, kernel_sizes, dense_layers, pooling_type, pooling_size, activation, optimizer)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return accuracy

# Function to build the model
def build_model(input_shape, n_classes, conv_layers, kernel_sizes, dense_layers, pooling_type='max', pooling_size=1, activation='relu', optimizer='adam'):
    model = tf.keras.Sequential()
    for i, (filters, kernel_size) in enumerate(zip(conv_layers, kernel_sizes)):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation, input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation))
        
        # Add pooling layer
        if pooling_type == 'max':
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(pooling_size, pooling_size)))
        elif pooling_type == 'average':
            model.add(tf.keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size)))
    
    model.add(tf.keras.layers.Flatten())
    
    for units in dense_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best parameters
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Use the best parameters to build the final model
best_params = trial.params
conv_layers = [best_params[f'n_filters_{i}'] for i in range(best_params['n_conv_layers'])]
kernel_sizes = [best_params[f'kernel_size_{i}'] for i in range(best_params['n_conv_layers'])]
dense_layers = [best_params[f'n_units_{i}'] for i in range(best_params['n_dense_layers'])]
model = build_model((1, 1, n_bands), n_classes, conv_layers, kernel_sizes, dense_layers, best_params['pooling_type'], best_params['pooling_size'], best_params['activation'], best_params['optimizer'])

# Train the final model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the final model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict the labels for the entire image
X_all = np.moveaxis(hyperspectral_data, 0, -1)  # Shape: (n_rows, n_cols, n_bands)
X_all_flat = X_all.reshape(-1, n_bands)
X_all_flat = scaler.transform(X_all_flat)
X_all = X_all_flat.reshape(-1, 1, 1, n_bands)

y_pred = model.predict(X_all)
y_pred = np.argmax(y_pred, axis=1)
y_pred = np.array([inverse_label_mapping[idx] for idx in y_pred])
y_pred = y_pred.reshape((n_rows, n_cols))

# Prepare the mask for correct/incorrect predictions
y_true_full = land_cover_labels
correct_mask = (y_pred == y_true_full)

# Visualize the predicted labels and the correct/incorrect predictions
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Predicted labels
im0 = axes[0].imshow(y_pred, cmap='tab20')
axes[0].set_title("Predicted Land Cover Labels")
axes[0].set_xlabel("Columns")
axes[0].set_ylabel("Rows")
fig.colorbar(im0, ax=axes[0])

# Correct vs Incorrect
correct_incorrect_plot = np.zeros_like(y_true_full, dtype=np.float32)
correct_incorrect_plot[correct_mask] = 1  # Correct predictions
correct_incorrect_plot[~correct_mask] = -1  # Incorrect predictions

im = axes[1].imshow(correct_incorrect_plot, cmap='bwr', vmin=-1, vmax=1)
axes[1].set_title("Correct (blue) vs Incorrect (red) Predictions")
axes[1].set_xlabel("Columns")
axes[1].set_ylabel("Rows")
fig.colorbar(im, ax=axes[1], orientation='vertical', label='Prediction Accuracy')

plt.tight_layout()
plt.show()