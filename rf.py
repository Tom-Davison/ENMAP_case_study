import xml.etree.ElementTree as ET
import rasterio
import numpy as np
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

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
if crs != wc_crs:
    transformer = Transformer.from_crs('EPSG:4326', wc_crs, always_xy=True)
    coords_wc = np.array([[transformer.transform(lon, lat) for lon, lat in row] for row in geographic_coords])
else:
    coords_wc = geographic_coords

# Visualize the overlay of EnMAP on ESA WorldCover
fig, ax = plt.subplots(figsize=(10, 10))

# Plot ESA WorldCover data
img_wc = ax.imshow(worldcover_data, cmap='tab20', extent=[wc_transform[2], wc_transform[2] + wc_transform[0] * worldcover_data.shape[1],
                                                          wc_transform[5] + wc_transform[4] * worldcover_data.shape[0], wc_transform[5]])

# Overlay EnMAP first band data
extent = [coords_wc[0, 0, 0], coords_wc[0, -1, 0], coords_wc[-1, 0, 1], coords_wc[0, 0, 1]]
img_enmap = ax.imshow(hyperspectral_data[0, :, :], cmap='gray', alpha=0.75, extent=extent, vmin=-10000, vmax=12000)

ax.set_title('EnMAP Overlay on ESA WorldCover')
plt.colorbar(img_wc, ax=ax, orientation='vertical', label='ESA WorldCover Classes')
plt.colorbar(img_enmap, ax=ax, orientation='horizontal', label='EnMAP First Band', pad=0.1)
plt.show()


# Ensure both CRS are the same; reproject if necessary
if crs != wc_crs:
    transformer = Transformer.from_crs(crs, wc_crs, always_xy=True)
    coords_wc = np.array([[transformer.transform(lon, lat) for lon, lat in row] for row in geographic_coords])
else:
    coords_wc = geographic_coords

def get_worldcover_label(lat, lon):
    # Convert lat/lon to row/col in WorldCover raster
    print("Lat/Lon:", lat, lon)
    row, col = ~wc_transform * (lon, lat)
    print("Row/Col:", row, col)
    row, col = int(row), int(col)
    print(worldcover_data.shape[0], worldcover_data.shape[1])
    print("")
    if 0 <= row < worldcover_data.shape[0] and 0 <= col < worldcover_data.shape[1]:
        return worldcover_data[row, col]
    else:
        return None

# Get land cover labels for all coordinates
land_cover_labels = np.array([[get_worldcover_label(lat, lon) for lon, lat in row] for row in coords_wc])
print("Land Cover Labels:", land_cover_labels)
exit()

# Normalize the data
reshaped_data = hyperspectral_data.reshape(n_bands, n_rows * n_cols).T
scaler = StandardScaler()
normalized_data = scaler.fit_transform(reshaped_data)

# Reshape land cover labels to match the data
labels = land_cover_labels.flatten()

# Filter out pixels with no land cover label (if any)
valid_mask = labels != None
X = normalized_data[valid_mask]
y = labels[valid_mask]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Validate the model
scores = cross_val_score(rf, X, y, cv=5)
print('Cross-validation scores:', scores)

# Predict land cover types for the hyperspectral data
land_type_predictions = rf.predict(normalized_data)

# Reshape predictions back to image dimensions
land_type_image = land_type_predictions.reshape(n_rows, n_cols)

# Visualize the land type classification
fig, ax = plt.subplots(figsize=(10, 10))
img = ax.imshow(land_type_image, cmap='tab20')
ax.set_title('Land Type Classification')

# Set coordinate ticks

plt.colorbar(img, ax=ax)
plt.show()