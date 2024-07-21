import rasterio
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
import matplotlib.colors as mcolors

def read_files(enmap_data_path, enmap_metadata_path, esa_worldcover_path, class_mapping):
    

    # Define the class mapping for
    cmap = plt.cm.get_cmap('Accent', len(class_mapping))
    norm = mcolors.BoundaryNorm(boundaries=[key - 0.5 for key in sorted(class_mapping.keys())] + [max(class_mapping.keys()) + 0.5], ncolors=len(class_mapping))

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
        boundary_coords_wc = np.array([[transformer.transform(lon, lat) for lat, lon in row] for row in coordinates])
    else:
        coords_wc = geographic_coords
        boundary_coords_wc = coordinates

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
    """
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
    """

    # Create a polygon from the transformed coordinates
    print(boundary_coords_wc)
    #transform boundary_coords to a list of tuples for Polygon
    boundary_coords_wc = [(lon, lat) for lat, lon in boundary_coords_wc]
    print(boundary_coords_wc)

    polygon = Polygon(boundary_coords_wc)
    print(polygon)
    geojson_polygon = [mapping(polygon)]
    #exit()
    # Clip the WorldCover data to the bounding polygon
    with rasterio.open(esa_worldcover_path) as src:
        out_image, out_transform = mask(src, geojson_polygon, crop=True)

    # Calculate the extents for the clipped WorldCover data
    clipped_wc_extent = [
        out_transform[2], out_transform[2] + out_transform[0] * out_image.shape[2],
        out_transform[5] + out_transform[4] * out_image.shape[1], out_transform[5]
    ]
    """
    # Visualize the overlay of EnMAP on the clipped ESA WorldCover
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot clipped ESA WorldCover data
    img_wc_clipped = ax.imshow(out_image[0], cmap='tab20', extent=clipped_wc_extent)

    # Overlay EnMAP first band data
    img_enmap_clipped = ax.imshow(hyperspectral_data[0, :, :], cmap='gray', alpha=0.75, extent=enmap_extent, vmin=np.nanmin(hyperspectral_data[0, :, :]), vmax=np.nanmax(hyperspectral_data[0, :, :]))

    ax.set_title('EnMAP Overlay on Clipped ESA WorldCover')
    plt.colorbar(img_wc_clipped, ax=ax, orientation='vertical', label='Clipped ESA WorldCover Classes')
    plt.colorbar(img_enmap_clipped, ax=ax, orientation='horizontal', label='EnMAP First Band', pad=0.1)
    plt.show()
    """
    # Get land cover labels for all coordinates
    land_cover_labels = np.array([[get_worldcover_label(lat, lon) for lon, lat in row] for row in coords_wc])
    print("Land Cover Labels:", land_cover_labels)


    # Prepare the data
    # Assuming hyperspectral_data and land_cover_labels are already defined
    n_bands, n_rows, n_cols = hyperspectral_data.shape
    X = hyperspectral_data # np.moveaxis(hyperspectral_data, 0, -1)  # Shape: (n_rows, n_cols, n_bands)
    y = land_cover_labels

    # Remove pixels with None labels
    #mask = y != None
    #X = X[mask]
    #y = y[mask]

    # Flatten the 2D spatial dimensions to 1D for CNN input
    #X = X.reshape(-1, n_bands)
    #y = y.flatten()
    # Map labels to a range [0, n_classes-1]

    X = X.transpose(1, 2, 0)
    y = (y / 10) - 1
    y = y.astype(int)
    print(X.shape)
    print(y.shape)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # First image
    ground_truth = axes[0].imshow(y, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ground_truth, ax=axes[0], ticks=sorted(class_mapping.keys()))
    cbar.ax.set_yticklabels([class_mapping[key] for key in sorted(class_mapping.keys())])
    axes[0].set_title("Ground Truth")

    # Second image
    raw_data = axes[1].imshow(X[:, :, 0], cmap='gray', vmin = -100, vmax = 2700)
    #add colorbar
    cbar = plt.colorbar(raw_data, ax=axes[1])
    axes[1].set_title("Grayscale Image")

    plt.show()




    # Check unique labels and determine the number of classes
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    print(f"Unique labels: {unique_labels}")
    print(f"Number of classes: {n_classes}")