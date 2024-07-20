import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import time

# Configure Streamlit to use the full screen width
st.set_page_config(layout="wide")

@st.cache_data
def load_hyperspectral_data(file_path):
    with rasterio.open(file_path) as src:
        return src.read()
    
# Timing the loading process
start_time = time.time()

# Load the hyperspectral data
file_path = 'data/ENMAP01-____L2A-DT0000002446_20220810T112429Z_002_V010303_20230922T131813Z-SPECTRAL_IMAGE.tif'
hyperspectral_data = load_hyperspectral_data(file_path)

load_time = time.time() - start_time

# Get the number of bands
n_bands = hyperspectral_data.shape[0]

# Streamlit app
st.title('Hyperspectral Image Viewer')

# Create three columns for the sliders
col1, col2, col3 = st.columns(3)

# Timing the slider interaction process
start_time = time.time()

# Slider to select band
with col1:
    band = st.slider('Select Band', 1, n_bands, 1)

slider_time = time.time() - start_time

# Display the selected band
selected_band = hyperspectral_data[band - 1, :, :]

# Get the min and max for the selected band to set slider range
band_min, band_max = selected_band.min(), selected_band.max()

# Sliders to adjust vmin and vmax
start_time = time.time()

with col2:
    vmin = st.slider('Select Min Value', float(band_min), float(band_max), float(band_min))
with col3:
    vmax = st.slider('Select Max Value', float(band_min), float(band_max), float(band_max))

slider_minmax_time = time.time() - start_time

# Timing the plotting process
start_time = time.time()

# Plot the selected band with viridis color scheme and dynamic color scale
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust figure size for larger display
img = ax.imshow(selected_band, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title(f'Band {band}')
plt.colorbar(img, ax=ax)
st.pyplot(fig)

plot_time = time.time() - start_time
