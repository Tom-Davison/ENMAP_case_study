class_mapping = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built up",
    60: "bare/sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    100: "moss and lichen",
}

unit_class_mapping = {
    new_index: original_key
    for new_index, original_key in enumerate(class_mapping.keys())
}


num_components = 150
sample_cap = 300000


enmap_data = {
    "entry1": {
        "image": "data/enmap_image_austria.tif",
        "metadata": "data/enmap_metadata_austria.XML",
        "reference": "data/worldcover_austria.tif",
        "usage": "training",
        "area_code": "austria",
        "cluster": True,
    },
    "entry2": {
        "image": "data/enmap_image_nevada.TIF",
        "metadata": "data/enmap_metadata_nevada.XML",
        "reference": "data/worldcover_nevada.tif",
        "usage": "training",
        "area_code": "nevada",
        "cluster": False
    },
    "entry3": {
        "image": "data/enmap_image_holland.TIF",
        "metadata": "data/enmap_metadata_holland.XML",
        "reference": "data/worldcover_holland.tif",
        "usage": "training",
        "area_code": "holland",
        "cluster": False
    },
    "entry4": {
        "image": "data/enmap_image_spain.TIF",
        "metadata": "data/enmap_metadata_spain.XML",
        "reference": "data/worldcover_spain.tif",
        "usage": "training",
        "area_code": "spain",
        "cluster": False
    },
    "entry5": {
        "image": "data/enmap_image_spain2.TIF",
        "metadata": "data/enmap_metadata_spain2.XML",
        "reference": "data/worldcover_spain.tif",
        "usage": "testing",
        "area_code": "spain2",
        "cluster": False
    },
    "entry7": {
        "image": "data/enmap_image_brazil.TIF",
        "metadata": "data/enmap_metadata_brazil.XML",
        "reference": "data/worldcover_brazil.tif",
        "usage": "training",
        "area_code": "brazil",
        "cluster": False
    },
    "entry9": {
        "image": "data/enmap_image_florida2.TIF",
        "metadata": "data/enmap_metadata_florida2.XML",
        "reference": "data/worldcover_florida.tif",
        "usage": "training",
        "area_code": "florida2",
        "cluster": False
    },
    "entry10": {
        "image": "data/enmap_image_florida3.TIF",
        "metadata": "data/enmap_metadata_florida3.XML",
        "reference": "data/worldcover_florida.tif",
        "usage": "training",
        "area_code": "florida3",
        "cluster": False
    },
    "entry12": {
        "image": "data/enmap_image_tibet2.TIF",
        "metadata": "data/enmap_metadata_tibet2.XML",
        "reference": "data/worldcover_tibet2.tif",
        "usage": "training",
        "area_code": "tibet2",
        "cluster": False
    }
}

enmap_data_unuused = {
    "entry11": {
        "image": "data/enmap_image_tibet.TIF",
        "metadata": "data/enmap_metadata_tibet.XML",
        "reference": "data/worldcover_tibet2.tif",
        "usage": "training",
        "area_code": "tibet",
        "cluster": False
    },
    "entry13": {
        "image": "data/enmap_image_tibet3.TIF",
        "metadata": "data/enmap_metadata_tibet3.XML",
        "reference": "data/worldcover_tibet2.tif",
        "usage": "training",
        "area_code": "tibet3",
        "cluster": False
    },
    "entry14": {
        "image": "data/enmap_image_tibet4.TIF",
        "metadata": "data/enmap_metadata_tibet4.XML",
        "reference": "data/worldcover_tibet2.tif",
        "usage": "training",
        "area_code": "tibet4",
        "cluster": False
    },
    "entry15": {
        "image": "data/enmap_image_tibet5.TIF",
        "metadata": "data/enmap_metadata_tibet5.XML",
        "reference": "data/worldcover_tibet2.tif",
        "usage": "training",
        "area_code": "tibet5",
        "cluster": False
    },
    "entry8": {
        "image": "data/enmap_image_florida.TIF",
        "metadata": "data/enmap_metadata_florida.XML",
        "reference": "data/worldcover_florida2.tif",
        "usage": "training",
        "area_code": "florida",
        "cluster": False
    },
    "entry6": {
        "image": "data/enmap_image_spain3.TIF",
        "metadata": "data/enmap_metadata_spain3.XML",
        "reference": "data/worldcover_spain.tif",
        "usage": "training",
        "area_code": "spain3",
        "cluster": False
    }
}
