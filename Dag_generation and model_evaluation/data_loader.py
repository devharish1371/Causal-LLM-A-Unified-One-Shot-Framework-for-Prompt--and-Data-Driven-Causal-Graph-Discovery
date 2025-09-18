import pandas as pd

def load_data(ground_truth_path, gaussian_path=None, non_gaussian_path=None):
    ground_truth = pd.read_csv(ground_truth_path, header=None).values
    node_labels = pd.read_csv(ground_truth_path, header=0).columns.tolist()

    gaussian_samples = pd.read_csv(gaussian_path, header=0).to_numpy() if gaussian_path else None
    non_gaussian_samples = pd.read_csv(non_gaussian_path, header=0).to_numpy() if non_gaussian_path else None

    return ground_truth, gaussian_samples, non_gaussian_samples, node_labels
