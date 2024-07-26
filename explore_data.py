
from read_files import standardise_images
from process_files import prep_training_data, generate_decomposition
from CNN import train_test_CNN, predict_CNN
from cluster_tools import cluster_image
from case_studies import generate_case_1, generate_case_2

# Configuration dictionary
config = {
    "process_raw_files": False,
    "cluster_image": False,
    "generate_decomposition": False,
    "prepare_training_data": False,
    "train_model": False,
    "test_model": False,
    "run_case_study_1": False,
    "run_case_study_2": True
}

def read_files(plot=True):
    if config["process_raw_files"]:
        standardise_images(plot=plot)

def cluster_images():
    if config["cluster_image"]:
        cluster_image()

def generate_decomp():
    if config["generate_decomposition"]:
        generate_decomposition()

def train_model():
    if config["train_model"]:
        X_train, X_test, y_train, y_test = prep_training_data(regenerate_library=config["prepare_training_data"])
        train_test_CNN(X_train, y_train, X_test, y_test)

def test_model():
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