import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
from matplotlib.markers import MarkerStyle

import config

def case_1_plot():

    centroids = [
        (-61.07271140792883, -12.801677198081865),
        (-61.105703003772135, -14.204962617481586),
        (-61.48408175793223, -13.913225695674738),
        (-61.54466690852023, -14.183717760378533),
        (-61.60563973969831, -14.455465658341993),
        (-61.66674127979763, -14.727132848726232)   
        ]
    df_centroids = pd.DataFrame(centroids, columns=['lon', 'lat'])
    st.map(data=df_centroids, latitude='lat', longitude='lon')

    area_codes = []
    for paths in config.enmap_data.values():
        if paths["usage"] == "case_study_1":
            area_codes.append(paths["area_code"])
    
    st.write(
    """
    Here we view EnMAP data taken from both Brazil and Bolivia. According to the Worldcover data, the area is entirely forested.
    However subsequent to that time (2021) the Brazilian government has allowed deforestation to occur in the area. The Bolivian forrest
    on the other hand has been protected within the national park. We are able to see this using the classified data, where the first image 
    has a large part of 'shrubland' indicating a loss of forrest; whereas Bolivian images (all others) show a retention of forrest.
    """)

    st.write(
    """
    Note that matplotlib allocated a colour to a displayed pixel based on all pixels within the original resolution. As such, the image can be misleading
    and the 'Class Fractions' plot should be interperted as the true representation of the data.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Raw spectral data (averaged)')
    with col2:
        st.write('Classified image')
    with col3:
        st.write('Class fractions')

    
    for area_code in area_codes:        
        data = np.load(f'data/streamlit/case1_{area_code}_data.npy', allow_pickle=True).item()

        col1, col2, col3 = st.columns(3)
        
        with col1:
            vmin, vmax = np.nanpercentile(data['raw_spectral'], [2, 98])
            fig, ax = plt.subplots()
            im = ax.imshow(data['raw_spectral'], vmin=max(vmin, -1000), vmax=vmax)
            cbar = plt.colorbar(im)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            im = ax.imshow(data['classified_image'], cmap=cmap, vmin=0, vmax=10)
            cbar = plt.colorbar(im)
            cbar.set_ticks(midpoints)
            cbar.set_ticklabels(config.short_class_mapping.values())
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots()
            ax.bar(data['class_names'], data['class_fractions'], color=[x for x in config.value_to_color_maps.values()])
            plt.xticks(rotation=90)
            st.pyplot(fig)

st.set_page_config(layout="wide", page_title="Case Study: Deforestation", page_icon=":deciduous_tree:",)
st.title("Case Study: Deforestation")
colors = list(config.value_to_color_maps.values())
values = list(config.value_to_color_maps.keys())
cmap = ListedColormap(colors)
midpoints = np.linspace(0.5, 9.5, 10)
case_1_plot()