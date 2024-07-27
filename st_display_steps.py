import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
from rasterio.transform import Affine
import rasterio
import streamlit as st
import json

import config

@st.cache_data
def load_saved_data(area_code):
    # Load all numpy arrays from a single file
    with np.load(f"data/streamlit/{area_code}_data.npz") as data:
        enmap_avg = data['enmap_avg']
        wc_image = data['wc_image']
        label_array = data['label_array']
        valid_mask = data['valid_mask']
    
    # Load metadata, plot configurations, data from a single JSON file
    with open(f"data/streamlit/{area_code}_info.json", "r") as f:
        info = json.load(f)
    
    metadata = info['metadata']
    plot_configs = info['plot_configs']
    
    return enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs


def recreate_plots(enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs):
    enmap_transform = Affine.from_gdal(*metadata['enmap_transform'])
    wc_transform = Affine.from_gdal(*metadata['wc_transform'])
    color_scale = metadata['enmap_color_scale']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    plot_enmap(ax1, enmap_avg, valid_mask, color_scale, plot_configs['original_enmap']['title'], enmap_transform)
    plot_overlay(ax2, enmap_avg, wc_image, valid_mask, enmap_transform, wc_transform, color_scale)
    plot_labels(ax3, enmap_avg, label_array, valid_mask, plot_configs['labels']['title'], enmap_transform)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_enmap(ax, enmap_avg, valid_mask, color_scale, title, transform):
    masked_enmap = np.ma.array(enmap_avg, mask=~valid_mask)
    height, width = enmap_avg.shape
    extent = rasterio.transform.array_bounds(height, width, transform)

    im = ax.imshow(masked_enmap, cmap='viridis', vmin=color_scale[0], vmax=color_scale[1], extent=extent)
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Averaged Spectral Value')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def plot_overlay(ax, enmap_avg, wc_image, valid_mask, enmap_transform, wc_transform, color_scale):
    masked_enmap = np.ma.array(enmap_avg, mask=~valid_mask)

    enmap_extent = rasterio.transform.array_bounds(enmap_avg.shape[0], enmap_avg.shape[1], enmap_transform)
    wc_extent = rasterio.transform.array_bounds(wc_image.shape[0], wc_image.shape[1], wc_transform)

    enmap_plot = ax.imshow(masked_enmap, cmap="viridis", extent=enmap_extent, alpha=0.7, vmin=color_scale[0], vmax=color_scale[1])
    ax.imshow(wc_image, cmap="tab10", extent=wc_extent, alpha=0.3)

    ax.set_title("EnMAP and ESA WorldCover Data Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(enmap_plot, cax=cax, label="EnMAP Reflectance")

def plot_labels(ax, enmap_avg, label_array, valid_mask, title, transform):
    height, width = enmap_avg.shape
    extent = rasterio.transform.array_bounds(height, width, transform)
    
    mask = (label_array == 0) | ~valid_mask
    masked_labels = np.ma.array(label_array, mask=mask)
    
    label_plot = ax.imshow(masked_labels, cmap=cmap, norm=norm, alpha=0.5, extent=extent)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(label_plot, cax=cax)
    cbar.set_label('Land Cover Classes')
    
    cbar.set_ticks(midpoints)
    cbar.set_ticklabels([config.class_mapping[key] for key in values])
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


st.set_page_config(layout="wide")
st.title("EnMAP Data Viewer")

# Extract colors and values
colors = list(config.value_to_color_maps.values())
values = list(config.value_to_color_maps.keys())
cmap = ListedColormap(colors)
boundaries = values + [values[-1] + 1]
norm = BoundaryNorm(boundaries, cmap.N)
midpoints = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]


# Extract all area_codes from the config
area_codes = [entry['area_code'] for entry in config.enmap_data.values() if 'area_code' in entry]

# Create a dropdown to select the area_code
selected_area_code = st.selectbox("Choose an area", area_codes)

# Get the data for the selected area_code
selected_data = next((entry for entry in config.enmap_data.values() if entry['area_code'] == selected_area_code), None)

if selected_data:
    enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs = load_saved_data(selected_area_code)
    recreate_plots(enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs)
else:
    st.error(f"No data found for area code: {selected_area_code}")