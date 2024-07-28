
from read_files import standardise_images
from process_files import prep_training_data, generate_decomposition
from CNN import train_test_CNN, predict_CNN
from cluster_tools import cluster_image
from case_studies import generate_case_1, generate_case_2

# Choose coponents to run
config = {
    "process_raw_files": False,
    "cluster_image": False,
    "generate_decomposition": False,
    "prepare_training_data": True,
    "train_model": True,
    "test_model": True,
    "run_case_study_1": True,
    "run_case_study_2": True
}

def read_files(plot=False):
    # Read and standardise the images ensuring EnMAP and ESA WorldCover data are 
    # in the same format and projection. This will be the basis for our input and
    # laballed data for the CNN model.
    if config["process_raw_files"]:
        standardise_images(plot=plot)

def cluster_images():
    # Cluster the images using KMeans to investigate the spectral variance. This is
    # compared to the ESA WorldCover data to evaluate the spectral clustering vs human
    # oriented classification.
    if config["cluster_image"]:
        cluster_image()

def generate_decomp():
    # Build a sample of images to generate a balanced dataset for decomposition. Balancing 
    # attempts to ensure that the decomposition model is not biased towards the most common 
    # classes. This is used to train a FastICA decomposition model to reduce dimensionality 
    # prior to training the CNN model.
    if config["generate_decomposition"]:
        generate_decomposition()

def train_model():
    # Prepare the training data for the CNN model and train the model. The model is then
    # tested on labelled data to evaluate the performance. The model can be tuned to optimise
    # performance. The output model attempts to classify the land type of the input images, 
    # using ESA worldcover data as the ground truth.
    if config["train_model"]:
        X_train, X_test, y_train, y_test = prep_training_data(regenerate_library=config["prepare_training_data"])
        train_test_CNN(X_train, y_train, X_test, y_test, tune=False)

def test_model():
    # Label an EnMAP image using the trained CNN model. This can be used to evaluate the model
    # and perfomance metrics are provided
    if config["test_model"]:
        predict_CNN()

def case_studies():
    if config["run_case_study_1"]:
        generate_case_1()
    if config["run_case_study_2"]:
        generate_case_2()

def main():
    read_files()
    cluster_images()
    generate_decomp()
    train_model()
    test_model()
    case_studies()

if __name__ == "__main__":
    main()