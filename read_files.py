import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import xml.etree.ElementTree as ET

def read_files(enmap_data_path, enmap_metadata_path, esa_worldcover_path, plot=False):
    """
    Reads EnMAP data, EnMAP metadata, and ESA WorldCover data, reprojects both to EPSG:4326,
    and plots them overlaid.

    Parameters:
    - enmap_data_path: str, path to the EnMAP image data file.
    - enmap_metadata_path: str, path to the EnMAP metadata file.
    - esa_worldcover_path: str, path to the ESA WorldCover data file.
    - class_mapping: dict, a dictionary for class mapping (not used in this function).
    """

    target_crs = 'EPSG:4326'  # Define the target CRS (WGS 84)

    #read XML for boundary
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

    # Step 1: Read EnMAP Data
    with rasterio.open(enmap_data_path) as enmap_src:
        enmap_image = enmap_src.read()  # Read all bands
        enmap_transform = enmap_src.transform
        enmap_crs = enmap_src.crs
        print(f"CRS of the EnMAP data: {enmap_crs}")
        
        # Reproject EnMAP data to target CRS
        transform, width, height = calculate_default_transform(
            enmap_crs, target_crs, enmap_src.width, enmap_src.height, *enmap_src.bounds)
        
        enmap_image_reprojected = np.empty((enmap_image.shape[0], height, width), dtype=enmap_image.dtype)
        
        for i in range(enmap_image.shape[0]):
            reproject(
                source=enmap_image[i],
                destination=enmap_image_reprojected[i],
                src_transform=enmap_transform,
                src_crs=enmap_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest)
        
        enmap_image = enmap_image_reprojected
        enmap_transform = transform

    # Step 2: Read ESA WorldCover Data and clip to EnMAP extent
    with rasterio.open(esa_worldcover_path) as wc_src:
        wc_image = wc_src.read(1)  # Read the first band
        wc_transform = wc_src.transform
        wc_crs = wc_src.crs
        print(f"CRS of the WorldCover data: {wc_crs}")

        # Reproject WorldCover data to target CRS
        transform, width, height = calculate_default_transform(
            wc_crs, target_crs, wc_src.width, wc_src.height, *wc_src.bounds)

        wc_image_reprojected = np.empty((height, width), dtype=wc_image.dtype)

        reproject(
            source=wc_image,
            destination=wc_image_reprojected,
            src_transform=wc_transform,
            src_crs=wc_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest)

        wc_image = wc_image_reprojected
        wc_transform = transform

        # Clip WorldCover data to the EnMAP extent
        enmap_bounds = rasterio.transform.array_bounds(enmap_image.shape[1], enmap_image.shape[2], enmap_transform)
        enmap_bbox = box(*enmap_bounds)
        enmap_geo = gpd.GeoDataFrame({'geometry': [enmap_bbox]}, crs=target_crs)

        wc_out_image, wc_out_transform = mask(dataset=wc_src, shapes=enmap_geo.geometry, crop=True, all_touched=True)
        wc_out_image = wc_out_image[0]  # Extract the single band from the masked result

    # Calculate extents manually
    def calculate_extent(transform, width, height):
        """Calculate the extent (left, right, bottom, top) of the image based on the transform."""
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        return (left, right, bottom, top)

    enmap_extent = calculate_extent(enmap_transform, enmap_image.shape[2], enmap_image.shape[1])
    wc_extent = calculate_extent(wc_out_transform, wc_out_image.shape[1], wc_out_image.shape[0])

    if plot:
        # Step 3: Plot the Data Overlaid
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot EnMAP data
        enmap_plot = ax.imshow(enmap_image[0], cmap='gray', extent=enmap_extent, alpha=0.7, vmin=-50, vmax=3000)

        # Plot WorldCover data, limited to the EnMAP extent
        wc_plot = ax.imshow(wc_out_image, cmap='tab10', extent=wc_extent, alpha=0.3)

        # Add colorbars
        fig.colorbar(enmap_plot, ax=ax, fraction=0.036, pad=0.04, label='EnMAP Reflectance')
        fig.colorbar(wc_plot, ax=ax, fraction=0.036, pad=0.04, label='WorldCover Classification')

        # Set plot title and labels
        ax.set_title('EnMAP and ESA WorldCover Data Overlay')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Show plot
        plt.show()

    # Step 4: Generate Label Array from ESA WorldCover Data
    # Create an empty array for the labels with the same spatial dimensions as EnMAP
    label_array = np.empty((enmap_image.shape[1], enmap_image.shape[2]), dtype=wc_out_image.dtype)

    # Transform pixel coordinates in EnMAP to geographic coordinates
    for row in range(enmap_image.shape[1]):
        for col in range(enmap_image.shape[2]):
            # Get the geographic coordinates (latitude, longitude) of the EnMAP pixel
            x, y = enmap_transform * (col, row)
            
            # Transform the geographic coordinates to pixel coordinates in the WorldCover image
            wc_col, wc_row = ~wc_out_transform * (x, y)
            wc_col, wc_row = int(wc_col), int(wc_row)
            
            # Ensure indices are within the bounds of the WorldCover image
            if 0 <= wc_row < wc_out_image.shape[0] and 0 <= wc_col < wc_out_image.shape[1]:
                # Assign the label from the WorldCover image to the corresponding pixel in the label array
                label_array[row, col] = wc_out_image[wc_row, wc_col]
            else:
                label_array[row, col] = -1  # Assign a default value for out-of-bounds (e.g., -1 for no data)

    if plot:
        # Plot the new y data
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot EnMAP data for reference
        ax.imshow(enmap_image[0], cmap='gray', extent=enmap_extent, alpha=0.7)

        # Plot the new y data
        y_plot = ax.imshow(label_array, cmap='tab10', extent=enmap_extent, alpha=0.3)

        # Add colorbar
        fig.colorbar(y_plot, ax=ax, fraction=0.036, pad=0.04, label='New ESA Labels')

        # Set plot title and labels
        ax.set_title('New ESA Labels Overlay on EnMAP Data')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Show plot
        plt.show()

    y = label_array
    #y = (y / 10) - 1
    #y = y.astype(int)

    X = enmap_image  # Spectral data
    X = X.transpose(1, 2, 0)
    print("Spectral data shape:", X.shape)
    print("Label data shape:", y.shape)
    
    return X, y
