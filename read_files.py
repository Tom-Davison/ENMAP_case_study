import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from rasterio.mask import mask
from shapely.geometry import Polygon
import geopandas as gpd
import xml.etree.ElementTree as ET
from rasterio.plot import show
import config

def standardise_images(plot=False):
    for entry_name, paths in config.enmap_data.items():
        image_path = paths["image"]
        metadata_path = paths["metadata"]
        reference_path = paths["reference"]
        area_code = paths["area_code"]

        print(f"Processing {entry_name}:")
        read_and_convert_files(image_path, metadata_path, reference_path, area_code, plot=plot)

def save_arrays(X, y, base_filename):
    np.savez(f"data/cleaned_{base_filename}.npz", X=X, y=y)

def load_arrays(base_filename):
    data = np.load(f"data/cleaned_{base_filename}.npz")
    X = data['X']
    y = data['y']
    return X, y

def read_and_convert_files(enmap_data_path, enmap_metadata_path, esa_worldcover_path, area_code, plot=False):
    target_crs = "EPSG:4326"  # Define the target CRS (WGS 84)

    # Read XML for boundary
    tree = ET.parse(enmap_metadata_path)
    root = tree.getroot()

    # Extract bounding coordinates
    bounding_polygon = root.find(".//spatialCoverage/boundingPolygon")
    coordinates = [
        (float(point.find("longitude").text), float(point.find("latitude").text))
        for point in bounding_polygon.findall("point")
    ]

    # Convert coordinates to a Polygon
    bounding_polygon = Polygon(coordinates[0:5])
    gdf_polygon = gpd.GeoDataFrame([1], geometry=[bounding_polygon], crs=target_crs)

    # Step 1: Read and clip EnMAP Data
    with rasterio.open(enmap_data_path) as enmap_src:
        enmap_crs = enmap_src.crs
        gdf_polygon_enmap = gdf_polygon.to_crs(enmap_crs)

        enmap_image, enmap_transform = mask(
            dataset=enmap_src, shapes=gdf_polygon_enmap.geometry, crop=True
        )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        show(enmap_image[0], transform=enmap_transform, ax=ax, cmap="gray")
        gdf_polygon_enmap.boundary.plot(ax=ax, edgecolor="red")
        plt.title("Original EnMAP data with clipping polygon")
        plt.show()

    # Step 2: Reproject the clipped EnMAP data to EPSG:4326
    dst_crs = "EPSG:4326"
    transform, width, height = calculate_default_transform(
        enmap_crs,
        dst_crs,
        enmap_image.shape[2],
        enmap_image.shape[1],
        *rasterio.transform.array_bounds(
            enmap_image.shape[1], enmap_image.shape[2], enmap_transform
        ),
    )

    enmap_reprojected = np.empty(
        (enmap_image.shape[0], height, width), dtype=enmap_image.dtype
    )

    for i in range(enmap_image.shape[0]):
        reproject(
            source=enmap_image[i],
            destination=enmap_reprojected[i],
            src_transform=enmap_transform,
            src_crs=enmap_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    enmap_image = enmap_reprojected
    enmap_transform = transform

    if plot:
        gdf_polygon_reproj = gdf_polygon.to_crs(dst_crs)

        fig, ax = plt.subplots(figsize=(10, 10))
        show(enmap_image[0], transform=enmap_transform, ax=ax, cmap="gray")
        gdf_polygon_reproj.boundary.plot(ax=ax, edgecolor="red")
        plt.title("Reprojected EnMAP data with reprojected polygon")
        plt.show()

    # Step 3: Read and clip ESA WorldCover Data
    with rasterio.open(esa_worldcover_path) as wc_src:
        wc_crs = wc_src.crs
        gdf_polygon_wc = gdf_polygon.to_crs(wc_crs)

        wc_image, wc_transform = mask(
            dataset=wc_src, shapes=gdf_polygon_wc.geometry, crop=True
        )
        wc_image = wc_image[0]  # Extract the single band

    # Step 4: Reproject WorldCover data to target CRS
    wc_transform, wc_width, wc_height = calculate_default_transform(
        wc_crs,
        target_crs,
        wc_image.shape[1],
        wc_image.shape[0],
        *rasterio.transform.array_bounds(
            wc_image.shape[0], wc_image.shape[1], wc_transform
        ),
    )

    wc_reprojected = np.empty((wc_height, wc_width), dtype=wc_image.dtype)

    reproject(
        source=wc_image,
        destination=wc_reprojected,
        src_transform=wc_transform,
        src_crs=wc_crs,
        dst_transform=wc_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )

    wc_image = wc_reprojected

    if plot:
        plot_overlay(enmap_image, wc_image, enmap_transform, wc_transform)

    # Step 5: Generate Label Array from ESA WorldCover Data
    label_array, valid_mask = create_label_array(
        enmap_image, enmap_transform, dst_crs, wc_image, wc_transform, target_crs
    )

    if plot:
        plot_labels(enmap_image, label_array, enmap_transform)

    X = enmap_image.transpose(1, 2, 0)  # Spectral data
    y = label_array

    # Apply the mask to the EnMAP data and labels
    masked_X = np.where(valid_mask[:, :, np.newaxis], X, np.nan)
    masked_y = np.where(
        valid_mask, y, -1
    )  # Use -1 or another appropriate value for invalid pixels

    print("Original spectral data shape:", X.shape)
    print("Original label data shape:", y.shape)
    print("Masked spectral data shape:", masked_X.shape)
    print("Masked label data shape:", masked_y.shape)
    print(f"Total pixels: {X.shape[0] * X.shape[1]}")
    print(f"Valid pixels: {np.sum(valid_mask)}")

    save_arrays(masked_X, masked_y, area_code)


def create_label_array(
    enmap_image, enmap_transform, enmap_crs, wc_image, wc_transform, wc_crs
):
    # Create a mask for valid data in the WorldCover image (assuming 255 is no-data)
    wc_mask = wc_image != 255

    # Create an empty label array
    label_array = np.full(
        (enmap_image.shape[1], enmap_image.shape[2]), -1, dtype=np.int16
    )

    # Get the bounds of the EnMAP image
    enmap_bounds = rasterio.transform.array_bounds(
        enmap_image.shape[1], enmap_image.shape[2], enmap_transform
    )

    # Transform EnMAP bounds to WorldCover CRS
    wc_bounds = transform_bounds(enmap_crs, wc_crs, *enmap_bounds)

    # Calculate the range of pixels in the WorldCover image that correspond to the EnMAP extent
    wc_col_min, wc_row_min = ~wc_transform * (wc_bounds[0], wc_bounds[3])
    wc_col_max, wc_row_max = ~wc_transform * (wc_bounds[2], wc_bounds[1])

    wc_col_min, wc_col_max = max(0, int(min(wc_col_min, wc_col_max))), min(
        wc_image.shape[1] - 1, int(max(wc_col_min, wc_col_max))
    )
    wc_row_min, wc_row_max = max(0, int(min(wc_row_min, wc_row_max))), min(
        wc_image.shape[0] - 1, int(max(wc_row_min, wc_row_max))
    )

    # Iterate over the EnMAP pixels
    for row in range(enmap_image.shape[1]):
        for col in range(enmap_image.shape[2]):
            x, y = enmap_transform * (col, row)
            wc_col, wc_row = ~wc_transform * (x, y)
            wc_col, wc_row = int(wc_col), int(wc_row)

            if (
                wc_col_min <= wc_col <= wc_col_max
                and wc_row_min <= wc_row <= wc_row_max
                and wc_mask[wc_row, wc_col]
            ):
                label_array[row, col] = wc_image[wc_row, wc_col]

    valid_mask = label_array != -1
    return label_array, valid_mask


def plot_overlay(enmap_image, wc_image, enmap_transform, wc_transform):
    fig, ax = plt.subplots(figsize=(10, 10))
    enmap_extent = rasterio.transform.array_bounds(
        enmap_image.shape[1], enmap_image.shape[2], enmap_transform
    )
    wc_extent = rasterio.transform.array_bounds(
        wc_image.shape[0], wc_image.shape[1], wc_transform
    )

    enmap_plot = ax.imshow(
        enmap_image[0], cmap="gray", extent=enmap_extent, alpha=0.7, vmin=-200, vmax=300
    )
    wc_plot = ax.imshow(wc_image, cmap="tab10", extent=wc_extent, alpha=0.3)

    fig.colorbar(enmap_plot, ax=ax, fraction=0.036, pad=0.04, label="EnMAP Reflectance")
    fig.colorbar(
        wc_plot, ax=ax, fraction=0.036, pad=0.04, label="WorldCover Classification"
    )

    ax.set_title("EnMAP and ESA WorldCover Data Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def plot_labels(enmap_image, label_array, enmap_transform):
    fig, ax = plt.subplots(figsize=(10, 10))
    enmap_extent = rasterio.transform.array_bounds(
        enmap_image.shape[1], enmap_image.shape[2], enmap_transform
    )

    ax.imshow(
        enmap_image[0], cmap="gray", extent=enmap_extent, alpha=0.7, vmin=-200, vmax=300
    )
    y_plot = ax.imshow(label_array, cmap="tab10", extent=enmap_extent, alpha=0.3)

    fig.colorbar(y_plot, ax=ax, fraction=0.036, pad=0.04, label="New ESA Labels")

    ax.set_title("New ESA Labels Overlay on EnMAP Data")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
