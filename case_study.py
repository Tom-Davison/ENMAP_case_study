import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

# Load the hyperspectral data
file_path = 'data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-SPECTRAL_IMAGE.tif'
with rasterio.open(file_path) as src:
    hyperspectral_data = src.read()

# Get the number of bands
n_bands = hyperspectral_data.shape[0]

# Streamlit app
st.title('Hyperspectral Image Viewer')

# Create three columns for the sliders
col1, col2, col3 = st.columns(3)

# Slider to select band
with col1:
    band = st.slider('Select Band', 1, n_bands, 1)

# Display the selected band
selected_band = hyperspectral_data[band - 1, :, :]

# Get the min and max for the selected band to set slider range
band_min, band_max = selected_band.min(), selected_band.max()

# Sliders to adjust vmin and vmax
with col2:
    vmin = st.slider('Select Min Value', float(band_min), float(band_max), float(band_min))
with col3:
    vmax = st.slider('Select Max Value', float(band_min), float(band_max), float(band_max))

# Plot the selected band with viridis color scheme and dynamic color scale
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust figure size for larger display
ax.imshow(selected_band, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title(f'Band {band}')
plt.colorbar(ax.imshow(selected_band, cmap='viridis', vmin=vmin, vmax=vmax), ax=ax)
st.pyplot(fig)