# Hyperspectral Data Analysis and Modelling

This repository contains a python application for analysing EnMAP hyperspectral data using clustering algorithms. Following this an advanced CNN is built to classify the data into land types. All the steps can be viewed in a Streamlit dashboard for rapid and efficient visualisation.

## Features

- Load and align hyperspectral data, providing labels for each pixel
- Perform KMeans clustering and analyse the spectral signiatures
- Build, tune, and execute a CNN for land type classfication using ESA Worldcover data
- Test this model and apply to 2 real world cases

## Usage
Code:
1. Configure options in explore_data.py
2. Run explore_data.py
   ```sh
    python explore_data.py
    ```

Visuals:
1. Run the Streamlit application:
    ```sh
    streamlit run Data_and_Model.py
    ```
2. Open your web browser and navigate to `http://localhost:8501` to interact with the application.

![25c206cd6bfa43c2b676482faa80a3082aec5f6bee48aaf65809d519](https://github.com/user-attachments/assets/04e19b56-bb56-41de-890b-75f1b7fdb887)

![2857bcd056fd4cc1629ad2ddf96d9717adf7103a8cbd8756b7513fda](https://github.com/user-attachments/assets/82c01666-3b81-4c8a-b1eb-93c1cdc2b404)

![cc17799ecdb86a5d856310444d3ec063ef8de92d6cf5aff7075de4b8](https://github.com/user-attachments/assets/423029a1-c361-4212-8fdc-e7af41b50ca3)

![59fc6c4738fd99fd00c3e56a67eb72be2a0f573bf0b3801add7cd250](https://github.com/user-attachments/assets/f76dc64b-d624-4221-8952-3aaf24d4755c)
