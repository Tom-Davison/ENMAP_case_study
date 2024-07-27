import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split

import numpy as np
from rasterio.transform import Affine
import rasterio
import streamlit as st
import json
import plotly.graph_objects as go

import config


@st.cache_data
def load_saved_data(area_code):
    # Load all numpy arrays from a single file
    with np.load(f"data/streamlit/{area_code}_data.npz") as data:
        enmap_avg = data["enmap_avg"]
        wc_image = data["wc_image"]
        label_array = data["label_array"]
        valid_mask = data["valid_mask"]

    # Load metadata, plot configurations, data from a single JSON file
    with open(f"data/streamlit/{area_code}_info.json", "r") as f:
        info = json.load(f)

    metadata = info["metadata"]
    plot_configs = info["plot_configs"]

    return enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs

def spec_analysis():
    col1, col2, col3 = st.columns(3)

    data = np.load('data/streamlit/clustering_data.npz')
    with open('data/streamlit/cluster_metadata.json', 'r') as f:
        metadata = json.load(f)

    band_numbers = [int(label.split('_')[1]) for label in metadata['band_names']]

    with col1:
        fig, ax = plt.subplots()
        vmin, vmax = np.nanpercentile(data['raw_avg'], [2, 98])
        im = plt.imshow(data['raw_avg'], cmap='viridis', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('auto', adjustable='box')
        ax.set_title('Averaged EnMAP data before clustering')
        st.pyplot(fig)
    with col2:
        clustered_image = np.full((metadata['image_shape'][0] * metadata['image_shape'][1]), -1)  # -1 for masked areas
        valid_mask = data['valid_mask'].reshape(-1)
        clustered_image[valid_mask] = data['cluster_labels']
        clustered_image = clustered_image.reshape((metadata['image_shape'][0], metadata['image_shape'][1]))
        masked_clustered_image = np.ma.masked_where(clustered_image == -1, clustered_image)

        fig, ax = plt.subplots()
        im = ax.imshow(masked_clustered_image, cmap='tab10')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_position = (np.arange(10) + 0.5) * (masked_clustered_image.max() / 10)
        cbar.set_ticks(tick_position)
        cbar.set_ticklabels(np.arange(10))  # Set the labels to be 0-9
        cbar.set_label('Cluster ID')
        cbar.set_label('Cluster ID')
        ax.set_aspect('auto', adjustable='box')
        ax.set_title('Clustered EnMAP data')
        
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        print(data['accuracy_matrix'])
        print(data['accuracy_matrix'].shape)
        im = ax.imshow(data['accuracy_matrix'])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Accuracy Matrix')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('ESA Classification')
        ax.set_aspect('auto', adjustable='box')
        yticks = range(len([config.short_class_mapping[key] for key in values]))
        ax.set_yticks(yticks)
        ax.set_yticklabels([config.short_class_mapping[key] for key in values])
        st.pyplot(fig)

    st.subheader('Spectral Analysis')
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        for i, centroid in enumerate(data['centroids']):
            ax.plot(metadata['band_names'], centroid, label=f'Cluster {i}')
        ax.set_ylim(0, max(max(centroid) for centroid in data['centroids']) * 1.1)
        ax.set_title('Centroids of clusters')
        ax.set_xlabel('Band')
        ax.set_ylabel('Reflectance')
        ax.legend()
        ax.set_xticks(band_numbers[::len(band_numbers) // 10])  # Adjust the number of ticks as needed
        ax.set_xticklabels(band_numbers[::len(band_numbers) // 10], rotation=45, ha='right')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        plt.plot(band_numbers, data['spectral_variance'])
        plt.ylim(data['spectral_variance'][-1] * 0.95, max(data['spectral_variance']) * 1.05)
        plt.title('Spectral variance for all pixels')
        plt.xlabel('Band')
        plt.ylabel('Variance Magnitude')
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.imshow(data['spectral_angles'], cmap='viridis')
        plt.colorbar(label='Spectral Angle (radians)', ax=ax)
        plt.title('Spectral Angles Between Cluster Centroids')
        plt.xlabel('Cluster')
        plt.ylabel('Cluster')
        plt.xticks(np.arange(data['spectral_angles'].shape[0]))
        plt.yticks(np.arange(data['spectral_angles'].shape[1]))
        st.pyplot(fig)

    st.subheader('PCA visualization in 3 planes')
    col1, col2, col3 = st.columns(3)
    _, pca_sample, _, cluster_labels_sample = train_test_split(data['pca_result'], data['cluster_labels'], stratify=data['cluster_labels'], train_size=0.99, random_state=42)
    
    pca_sample = np.array(pca_sample) / 1000
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(pca_sample[:, 0], pca_sample[:, 1], c=cluster_labels_sample, cmap='viridis', s=2, alpha=0.8)
        ax.set_title('PC1 vs PC2')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_xlim(-60, 20)
        ax.set_ylim(-20, 70)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(pca_sample[:, 0], pca_sample[:, 2], c=cluster_labels_sample, cmap='viridis', s=2, alpha=0.8)
        ax.set_title('PC1 vs PC2')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC3')
        ax.set_xlim(-60, 30)
        ax.set_ylim(-20, 15)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(pca_sample[:, 1], pca_sample[:, 2], c=cluster_labels_sample, cmap='viridis', s=2, alpha=0.8)
        ax.set_title('PC1 vs PC2')
        ax.set_xlabel('PC2')
        ax.set_ylabel('PC3')
        ax.set_xlim(-20, 70)
        ax.set_ylim(-25, 15)
        st.pyplot(fig)
    

def recreate_plots(
    enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs
):
    enmap_transform = Affine.from_gdal(*metadata["enmap_transform"])
    wc_transform = Affine.from_gdal(*metadata["wc_transform"])
    color_scale = metadata["enmap_color_scale"]

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_enmap(
            enmap_avg,
            valid_mask,
            color_scale,
            enmap_transform,
        )
    with col2:
        plot_overlay(
            enmap_avg, wc_image, valid_mask, enmap_transform, wc_transform, color_scale
        )
    with col3:
        plot_labels(
            enmap_avg,
            label_array,
            valid_mask,
            enmap_transform,
        )

    plt.tight_layout()


def plot_enmap(enmap_avg, valid_mask, color_scale, transform):
    masked_enmap = np.ma.array(enmap_avg, mask=~valid_mask)
    height, width = enmap_avg.shape
    extent = rasterio.transform.array_bounds(height, width, transform)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        masked_enmap,
        cmap="viridis",
        vmin=max(color_scale[0], -1000),
        vmax=color_scale[1],
        extent=extent,
    )
    ax.set_title('EnMAP Data, avergaed across spectral bands') 

    ax.set_xlabel("Array geometry")
    ax.set_ylabel("Array geometry")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label="EnMAP Reflectance")
    st.pyplot(fig)


def plot_overlay(
    enmap_avg, wc_image, valid_mask, enmap_transform, wc_transform, color_scale
):
    masked_enmap = np.ma.array(enmap_avg, mask=~valid_mask)

    enmap_extent = rasterio.transform.array_bounds(
        enmap_avg.shape[0], enmap_avg.shape[1], enmap_transform
    )
    wc_extent = rasterio.transform.array_bounds(
        wc_image.shape[0], wc_image.shape[1], wc_transform
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    enmap_plot = ax.imshow(
        masked_enmap,
        cmap="viridis",
        extent=enmap_extent,
        alpha=0.7,
        vmin=max(color_scale[0], -1000),
        vmax=color_scale[1],
    )
    ax.imshow(wc_image, cmap="tab10", extent=wc_extent, alpha=0.3)

    ax.set_title("EnMAP and ESA Worldcover Data Overlayed")
    ax.set_xlabel("Array geometry")
    ax.set_ylabel("Array geometry")
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(enmap_plot, cax=cax, label="EnMAP Reflectance")
    st.pyplot(fig)


def plot_labels(enmap_avg, label_array, valid_mask, transform):
    height, width = enmap_avg.shape
    extent = rasterio.transform.array_bounds(height, width, transform)

    mask = (label_array == 0) | ~valid_mask
    masked_labels = np.ma.array(label_array, mask=mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    label_plot = ax.imshow(
        masked_labels, cmap=cmap, norm=norm, alpha=0.5, extent=extent
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(label_plot, cax=cax)
    #cbar.set_label("Land Cover Classes")

    cbar.set_ticks(midpoints)
    cbar.set_ticklabels([config.short_class_mapping[key] for key in values])

    ax.set_title('EnMAP data remapped to Worldcover labels')
    ax.set_xlabel("Array geometry")
    ax.set_ylabel("Array geometry")
    
    st.pyplot(fig)


st.set_page_config(layout="wide")
st.title("EnMAP Data Viewer")


# Extract colors and values
colors = list(config.value_to_color_maps.values())
values = list(config.value_to_color_maps.keys())
cmap = ListedColormap(colors)
boundaries = values + [values[-1] + 1]
norm = BoundaryNorm(boundaries, cmap.N)
midpoints = [
    (boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)
]


# Extract all area_codes from the config
area_codes = [
    entry["area_code"] for entry in config.enmap_data.values() if "area_code" in entry
]

# Create a dropdown to select the area_code
selected_area_code = st.selectbox("Choose an area", area_codes)

# Get the data for the selected area_code
selected_data = next(
    (
        entry
        for entry in config.enmap_data.values()
        if entry["area_code"] == selected_area_code
    ),
    None,
)

st.header("Step 1: Align data and assign labels")
st.write(
    """
    Our first step is to read in the EnMAP data. This is provided in sensor geometry 
    so we must first assign a standard CRS and project into these coordinates. By doing so we also 
    allow for the reference data (ESA worldcover data) to be aligned. By resampling the ESA data 
    according to each pixel in the EnMAP data, we generate a label set for use later.
    """
)
st.write(
    """
    Note that the enmap data is contrained by a bounding Polygon indicating the valid are, so may 
    be clipped compared to the full sensor size. EnMAP colour scales are automatically adjusted to 
    2nd and 98th percentiles of the data to show detail, however this can be disrupted by incorrect bounds 
    provided in the EnMAP metadata.
    """
)

if selected_data:
    enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs = (
        load_saved_data(selected_area_code)
    )
    recreate_plots(enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs)
else:
    st.error(f"No data found for area code: {selected_area_code}")

st.header('Step 2: Hyperspectral data analysis and clustering')
spec_analysis()