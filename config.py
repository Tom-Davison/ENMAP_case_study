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

enmap_data = {
    "entry1": {
        "image": "data/training_data/enmap_image_austria.tif",
        "metadata": "data/training_data/enmap_metadata_austria.XML",
        "reference": "data/training_data/worldcover_austria.tif",
        "usage": "training",
        "area_code": "austria",
        "cluster": True
    },
    "entry2": {
        "image": "data/training_data/enmap_image_nevada.TIF",
        "metadata": "data/training_data/enmap_metadata_nevada.XML",
        "reference": "data/training_data/worldcover_nevada.tif",
        "usage": "training",
        "area_code": "nevada",
        "cluster": False
    },
    "entry3": {
        "image": "data/training_data/enmap_image_holland.TIF",
        "metadata": "data/training_data/enmap_metadata_holland.XML",
        "reference": "data/training_data/worldcover_holland.tif",
        "usage": "testing",
        "area_code": "holland",
        "cluster": False
    }
}

numComponents = 30
windowSize = 1
testRatio = 0.25
PATCH_SIZE = 1