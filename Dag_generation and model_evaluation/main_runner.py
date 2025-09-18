import os
from data_loader import load_data
from models import initialize_models, train_models
from evaluation import evaluate_and_save_results
from convert_npy_to_csv import convert_npy_to_csv

def main(ground_truth_path, gaussian_path=None, non_gaussian_path=None, extra_adj_matrix_path=None):
    ground_truth_name = os.path.splitext(os.path.basename(ground_truth_path))[0]
    ground_truth, gaussian_samples, non_gaussian_samples, node_labels = load_data(
        ground_truth_path, gaussian_path, non_gaussian_path
    )

    extra_adj_matrix = None
    if extra_adj_matrix_path:
        extra_adj_matrix = pd.read_csv(extra_adj_matrix_path, header=None).values

    combined_results = {}
    use_suffix = gaussian_samples is not None and non_gaussian_samples is not None

    if gaussian_samples is not None:
        models_g = initialize_models(gaussian_samples.shape[1])
        combined_results['g'] = train_models(models_g, gaussian_samples, 'g', node_labels, ground_truth_name)

    if non_gaussian_samples is not None:
        models_ng = initialize_models(non_gaussian_samples.shape[1])
        combined_results['ng'] = train_models(models_ng, non_gaussian_samples, 'ng', node_labels, ground_truth_name)

    evaluate_and_save_results(ground_truth, combined_results, ground_truth_name, extra_adj_matrix, use_suffix)

