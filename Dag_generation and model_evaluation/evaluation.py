import numpy as np
import pandas as pd

def count_accuracy(B_true, B_est):
    d = B_true.shape[0]
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])

    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

def evaluate_and_save_results(ground_truth, combined_results, ground_truth_name, extra_adj_matrix=None, use_suffix=True):
    accuracy_outputs = []

    for dataset_type in combined_results:
        for model_name, adj_matrix in combined_results[dataset_type].items():
            accuracy = count_accuracy(ground_truth, adj_matrix)
            accuracy['Causal Model'] = f"{model_name}({dataset_type.upper()})" if use_suffix else model_name
            accuracy_outputs.append(accuracy)

    if extra_adj_matrix is not None:
        extra_accuracy = count_accuracy(ground_truth, extra_adj_matrix)
        extra_accuracy['Causal Model'] = "LLM"
        accuracy_outputs.append(extra_accuracy)

    accuracy_df = pd.DataFrame(accuracy_outputs)
    cols = ['Causal Model'] + [col for col in accuracy_df.columns if col != 'Causal Model']
    accuracy_df = accuracy_df[cols]
    accuracy_df.to_excel(f"{ground_truth_name}_combined_results.xlsx", index=False)
