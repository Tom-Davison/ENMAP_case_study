import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
from rasterio.transform import Affine

from read_files import plot_overlay
import config

def load_saved_data(area_code):
    # Load all numpy arrays from a single file
    with np.load(f"data/streamlit/{area_code}_data.npz") as data:
        enmap_avg = data['enmap_avg']
        wc_image = data['wc_image']
        label_array = data['label_array']
        valid_mask = data['valid_mask']
    
    # Load metadata and plot configurations from a single JSON file
    with open(f"data/streamlit/{area_code}_info.json", "r") as f:
        info = json.load(f)
    
    metadata = info['metadata']
    plot_configs = info['plot_configs']
    
    return enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs

def plot_enmap(ax, enmap_avg, valid_mask, color_scale, title):
    # Create a masked array
    masked_enmap = np.ma.array(enmap_avg, mask=~valid_mask)
    
    # Plot using the color scale
    im = ax.imshow(masked_enmap, cmap='viridis', vmin=color_scale[0], vmax=color_scale[1])
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Averaged Spectral Value')

def recreate_plots(enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs):
    # Recreate transform objects
    enmap_transform = Affine.from_gdal(*metadata['enmap_transform'])
    wc_transform = Affine.from_gdal(*metadata['wc_transform'])
    color_scale = metadata['enmap_color_scale']
    
    # Original EnMAP plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_enmap(ax, enmap_avg, valid_mask, color_scale, plot_configs['original_enmap']['title'])
    st.pyplot(fig)
    
    # Reprojected EnMAP plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_enmap(ax, enmap_avg, valid_mask, color_scale, plot_configs['reprojected_enmap']['title'])
    st.pyplot(fig)
    
    # Overlay plot
    fig = plot_overlay(enmap_avg, wc_image, enmap_transform, wc_transform)
    st.pyplot(fig)
    
    # Labels plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot EnMAP data
    plot_enmap(ax, enmap_avg, valid_mask, color_scale, "EnMAP data")
    
    # Create a colormap for the labels
    unique_labels = np.unique(label_array)
    n_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('Set3', n_labels)
    
    # Create a masked array for the labels
    masked_labels = np.ma.array(label_array, mask=~valid_mask)
    
    # Plot labels on top
    label_plot = ax.imshow(masked_labels, cmap=cmap, alpha=0.5, 
                           vmin=unique_labels.min(), vmax=unique_labels.max())
    
    # Add colorbar for labels
    cbar = plt.colorbar(label_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Land Cover Classes')
    
    ax.set_title(plot_configs['labels']['title'])
    st.pyplot(fig)

# Streamlit app
st.title("EnMAP Data Viewer")

# Extract all area_codes from the config
area_codes = [entry['area_code'] for entry in config.enmap_data.values() if 'area_code' in entry]

# Create a dropdown to select the area_code
selected_area_code = st.selectbox("Choose an area", area_codes)

# Get the data for the selected area_code
selected_data = next((entry for entry in config.enmap_data.values() if entry['area_code'] == selected_area_code), None)

if selected_data:
    st.write(f"Selected area: {selected_area_code}")
    st.write(f"Image path: {selected_data['image']}")
    st.write(f"Metadata path: {selected_data['metadata']}")
    st.write(f"Reference path: {selected_data['reference']}")
    
    # Load the saved data
    enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs = load_saved_data(selected_area_code)
    
    # Recreate and display the plots
    recreate_plots(enmap_avg, wc_image, label_array, valid_mask, metadata, plot_configs)
else:
    st.error(f"No data found for area code: {selected_area_code}")