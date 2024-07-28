import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
from matplotlib import colors as mcolors
import numpy as np
from rasterio.transform import Affine
import rasterio
import streamlit as st
import json
import joblib
import pandas as pd
from collections import Counter

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

@st.cache_data
def spec_analysis():

    st.write(
        """
        Firstly we run a simple KMeans clustering algorithm on the hyperspectral data. This is set to provide
        10 clusters, the same as the ESA WorldCover data. The difference between these is that the ESA data is 
        oriented at human classification, whereas the clustering is based on spectral similarity. We may find that
        e.g two crop types show much greater spectral difference than e.g 'built-up' and 'bare' land types.
        """
    )
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
    st.write("""
            The spectral analysis is used to evaluate the spectral variance of the pixels in the data. We can evaluate what features of the spectrum
            provide the most variance and how the clusters are separated in spectral space. The spectral angles between the cluster centroids are also
            calculated to evaluate the spectral similarity between the clusters.
             """)
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
    st.write("""
            By visualizing the PCA results in 3 planes we can evaluate the clustering in the reduced dimensionality space. This can help to identify
            how the clusters are separated and if there is overlap between the clusters. The PCA results are scaled by 1000 to make the plots easier to read.
            """)
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
    

def reorg_data():
    # Load the decomposition info
    with open('data/streamlit/decomposition_info.json', 'r') as f:
        decomp_info = json.load(f)

    short_class_mapping = config.short_class_mapping
    value_to_color_maps = config.value_to_color_maps

    def sort_and_filter_data(data_dict):
        sorted_items = sorted(
            [(int(k), v) for k, v in data_dict.items() if int(k) in short_class_mapping],
            key=lambda x: list(short_class_mapping.keys()).index(x[0])
        )
        return zip(*sorted_items)

    def create_bar_chart(data_dict, title):
        labels, values = sort_and_filter_data(data_dict)
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=[value_to_color_maps[label] for label in labels], width=5)
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Fraction")
        ax.set_xticks(labels)
        ax.set_xticklabels([short_class_mapping[label] for label in labels], rotation=45, ha='right')
        plt.tight_layout()
        return fig

    before_chart = create_bar_chart(decomp_info['label_fractions_before'], "Before Balancing")
    after_chart = create_bar_chart(decomp_info['label_fractions_after'], "After Balancing")

    return decomp_info, before_chart, after_chart

@st.cache_data
def data_prep():
    decomp_info, before_chart, after_chart = reorg_data()

    st.subheader("Rebalancing label fractions by sampling and augmentation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(before_chart)

    with col2:
        st.pyplot(after_chart)

    with col3:
        st.subheader("FastICA Decomposition Information")
        st.write(f"Number of components: {decomp_info['num_components']}")
        st.write(f"Number of iterations: {decomp_info['n_iter']}")
        st.write(f"Components shape: {decomp_info['components_shape']}")

@st.cache_data
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


@st.cache_data
def model_train():

    with open('data/streamlit/model_metrics.json', 'r') as f:
        metrics = json.load(f)

    short_class_names = config.short_class_mapping.values()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(metrics['history']['accuracy'], label='Train Accuracy')
        ax.plot(metrics['history']['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Model Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(metrics['history']['loss'], label='Train Loss')
        ax.plot(metrics['history']['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)


    with col3:
        fig, ax = plt.subplots()
        im = plt.imshow(metrics['confusion_matrix'])
        plt.colorbar(im, ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)

    st.subheader("Classification Report")

    cr = metrics['classification_report']

    class_names = []
    precisions = []
    recalls = []
    f1_scores = []

    for class_name, class_metrics in cr.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            class_names.append(class_name)
            precisions.append(class_metrics['precision'])
            recalls.append(class_metrics['recall'])
            f1_scores.append(class_metrics['f1-score'])
            

    col1, col2, col3, = st.columns(3)

    with col1:
        fig, ax1 = plt.subplots()
        ax1.bar(class_names, precisions, color='b')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision by Class')
        ax1.set_xticklabels(short_class_names, rotation=45)
        ax1.set_ylim(0, 1.05)
        st.pyplot(fig)
    with col2:
        fig, ax2 = plt.subplots()
        ax2.bar(class_names, recalls, color='g')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Recall')
        ax2.set_title('Recall by Class')
        ax2.set_xticklabels(short_class_names, rotation=45)
        ax2.set_ylim(0, 1.05)
        st.pyplot(fig)
    with col3:
        fig, ax3 = plt.subplots()
        ax3.bar(class_names, f1_scores, color='r')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('F1-score')
        ax3.set_title('F1-score by Class')
        ax3.set_xticklabels(short_class_names, rotation=45)
        ax3.set_ylim(0, 1.05)
        st.pyplot(fig)

    #st.write(f"Overall Accuracy: {cr['accuracy']:.2f}")

@st.cache_data
def test_model():
    col1, col2 = st.columns(2)
    streamlit_data = joblib.load('data/streamlit/cnn_test_results.pkl')

    # Unpack the data
    results = pd.DataFrame(streamlit_data['class_metrics'])
    cm = streamlit_data['confusion_matrix']
    balanced_acc = streamlit_data['balanced_accuracy']
    outputs = streamlit_data['predicted_outputs']
    valid_mask = streamlit_data['valid_mask']
    correct_incorrect = streamlit_data['correct_incorrect']

    with col1:
        masked_image = np.ma.masked_where(valid_mask == 0, outputs)
        masked_image = (masked_image * 10) + 10

        fig, ax = plt.subplots()
        im = ax.imshow(masked_image, cmap=cmap, vmax=110)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks(midpoints)
        cbar.set_ticklabels([config.short_class_mapping[key] for key in values])
        cbar.set_label('Class')
        ax.set_title('Predicted Land Cover Classes')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(correct_incorrect, cmap='RdYlGn')
        st.pyplot(fig)
    with col1:
        # Display metrics
        st.write("Performance Metrics per Class:")
        st.dataframe(results)
    with col2:
        # Display confusion matrix
        st.write("Confusion Matrix:")
        st.write(cm)
    

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


st.set_page_config(layout="wide", page_title="Data and Model", page_icon=":earth_africa:",)
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
st.write("TODO: Add x-y picker tool and allow pixel to be take, spectra to be shown, and label to be shown")
st.header('Step 2: Hyperspectral data analysis and clustering')
st.write(
    """
    In this step we analyse the hyperspectral data to identify clusters of similar pixels. This is 
    done using KMeans clustering, where each pixel is assigned to a cluster based on the spectral 
    similarity to the cluster centroid. The centroids are used to identify the spectral signature of 
    each cluster. The clustering is compared to the ESA WorldCover data to evaluate the spectral 
    clustering vs human oriented classification.
    """
)
spec_analysis()

st.header('Step 3: Data preparation')
data_prep()

st.header('Step 4: Model training')
model_train()

st.header('Step 5: Model testing')
test_model()