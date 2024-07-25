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

enmap_data = {
    "entry1": {
        "image": "data/enmap_image_austria.tif",
        "metadata": "data/enmap_metadata_austria.XML",
        "reference": "data/worldcover_austria.tif",
        "usage": "training",
        "area_code": "austria",
        "cluster": True
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
        "usage": "testing",
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
        "usage": "training",
        "area_code": "spain2",
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

num_components = 100
sample_cap = 100000
