import numpy as np
from copy import deepcopy
import pandas as pd
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
import GPy

np.random.seed(42)

def preprocess_adj_matrix(adj_matrix, threshold=0.5):
    return np.where(np.abs(adj_matrix) > threshold, 1, 0)

def generate_W(d=6, prob=0.5):
    """
    Generate a random weighted adjacency matrix for a directed acyclic graph (DAG).

    Parameters:
    - d (int): Number of nodes in the graph.
    - prob (float): Probability of an edge existing between two nodes.

    Returns:
    - np.ndarray: A d x d weighted adjacency matrix with edge weights from specified ranges.
    """
    g_random = np.triu(np.random.rand(d, d) < prob, k=1)  # Upper triangular with p = 0.5
    weights_pos = np.random.uniform(0.5, 2.0, size=(d, d))
    weights_neg = np.random.uniform(-2.0, -0.5, size=(d, d))
    weights = np.where(np.random.rand(d, d) < 0.5, weights_pos, weights_neg)
    W = g_random * weights
    return W

def gen_data_given_model(b, n_samples=5000, noise_type='lingam', permutate=True):
    n_vars = b.shape[0]
    c = np.zeros(n_vars)
    assert b.shape == (n_vars, n_vars), "b must be a square matrix with shape (n_vars, n_vars)."
    assert c.shape == (n_vars,), "c must have shape (n_vars,)."
    assert np.allclose(b, np.triu(b)), "b must be strictly upper triangular."
    assert np.sum(np.abs(np.diag(b))) == 0, "Diagonal of b must be zero."

    s = np.ones(n_vars)

    if noise_type == 'lingam':
        q = np.random.rand(n_vars) * 1.1 + 0.5
        q[q > 0.8] += 0.4
        ss = np.sign(np.random.randn(n_samples, n_vars)) * (np.abs(np.random.randn(n_samples, n_vars)) ** q)
        ss = ss / np.std(ss, axis=0) * s
    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s
    else:
        raise ValueError("Invalid noise type. Choose 'lingam' or 'gaussian'.")

    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

    b_ = deepcopy(b)
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs = xs[:, p]
        b_ = b_[p, :][:, p]
        c_ = c[p]

    return xs, b_, c_

def generate_datasets(node_counts, edge_probability, n_samples, noise_type, permutate=False):
    datasets = {}
    adjacency_matrices = {}

    for num_nodes in node_counts:
        print(f"\nGenerating data for {num_nodes} nodes...")

        b = generate_W(num_nodes, edge_probability)
        data, b_, c_ = gen_data_given_model(b, n_samples=n_samples, noise_type=noise_type, permutate=permutate)
        datasets[num_nodes] = data
        adjacency_matrices[num_nodes] = b_
        b_ = preprocess_adj_matrix(b_, threshold=0.5)

        adj_matrix_df = pd.DataFrame(b_)
        adj_matrix_df.to_csv(f"adj_matrix_{num_nodes}nodes_linear.csv", index=False, header=[f"Feature {i}" for i in range(adj_matrix_df.shape[1])])
        noise = 'non-gaussian' if noise_type == 'lingam' else 'gaussian'
        data_df = pd.DataFrame(data)
        data_df.to_csv(f"{noise}_{num_nodes}nodes.csv", index=False, header=[f"Feature {i}" for i in range(data_df.shape[1])])

        print(f"Data for {num_nodes} nodes generated with shape: {data.shape}")

    print("All CSV files saved successfully.")
    return datasets, adjacency_matrices

def replace_nan_with_mean(xs):
    """Replace NaN values in each column of xs with the mean of that column."""
    for i in range(xs.shape[1]):
        fill_value = np.nanmean(xs[:, i])
        xs[np.isnan(xs[:, i]), i] = fill_value
    return xs

def gen_data_given_model_2nd_order(b, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model with quadratic functions."""
    n_vars = b.shape[0]
    c = np.zeros(n_vars)
    s = np.ones(n_vars)
    # Generate disturbance variables
    if noise_type == 'lingam':
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] += 0.4

        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s

    xs = np.zeros((n_samples, n_vars))
    poly = PolynomialFeatures()
    newb = []

    for i in range(n_vars):
        xs[:, i] = ss[:, i] + c[i]
        col = b[i]
        col_false_true = np.abs(col) > 0.3
        len_parents = int(np.sum(col_false_true))
        if len_parents == 0:
            newb.append(np.zeros(n_vars))
            continue

        X_parents = xs[:, col_false_true]
        X_2nd = poly.fit_transform(X_parents)
        X_2nd = X_2nd[:, 1:]  # Remove bias term
        dd = X_2nd.shape[1]

        # Coefficient sampling
        U = np.zeros(dd)
        for j in range(dd):
            if np.random.rand() < 0.5:
                U[j] = 0
            else:
                U[j] = np.random.choice([-1, -0.5, 0.5, 1]) * np.random.uniform(0.5, 1)

        X_sum = np.sum(U * X_2nd, axis=1)
        xs[:, i] += X_sum

        # Remove zero-weight variables
        X_train_expand_names = poly.get_feature_names_out()[1:]  # Exclude the bias term
        new_reg_coeff = np.zeros(n_vars)

        cj = 0
        for ci in range(n_vars):
            if col_false_true[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):
                    if xxi in xxx and np.abs(U[iii]) > 0:
                        new_reg_coeff[ci] = 1.0
                        break
                cj += 1

        newb.append(new_reg_coeff)

    # Permute variables
    b_ = deepcopy(np.array(newb))
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs = xs[:, p]
        b_ = b_[p, :][:, p]
        c_ = c[p]

    # Replace NaN values with mean
    xs = replace_nan_with_mean(xs)

    return xs, b_, c_

def generate_datasets_quad(node_counts, edge_probability, n_samples, noise_type, permutate=False):
    datasets = {}
    adjacency_matrices = {}

    for num_nodes in node_counts:
        print(f"\nGenerating data for {num_nodes} nodes...")

        b = generate_W(num_nodes, edge_probability)
        data, b_, c_ = gen_data_given_model_2nd_order(b, n_samples=n_samples, noise_type=noise_type, permutate=permutate)
        datasets[num_nodes] = data
        adjacency_matrices[num_nodes] = b_
        b_ = preprocess_adj_matrix(b_, threshold=0.5)

        adj_matrix_df = pd.DataFrame(b_)
        adj_matrix_df.to_csv(f"adj_matrix_{num_nodes}nodes_non-linear_quad.csv", index=False, header=[f"Feature {i}" for i in range(adj_matrix_df.shape[1])])
        noise = 'non-gaussian' if noise_type == 'lingam' else 'gaussian'
        data_df = pd.DataFrame(data)
        data_df.to_csv(f"{noise}_{num_nodes}nodes_non-linear_quad.csv", index=False, header=[f"Feature {i}" for i in range(data_df.shape[1])])

        print(f"Data for {num_nodes} nodes generated with shape: {data.shape}")

    print("All CSV files saved successfully.")
    return datasets, adjacency_matrices

def generate_graph_with_edges(d=6, prob=0.5):
    """
    Generate a random directed acyclic graph (DAG) represented as a NetworkX graph.

    Parameters:
        d (int): Number of nodes in the graph.
        prob (float): Probability of forming an edge.

    Returns:
        nx.DiGraph: A directed acyclic graph (DAG) object with edge attributes.
    """
    g_random = np.triu(np.random.rand(d, d) < prob, k=1)  # Upper triangular with probability
    W_binary = np.where(g_random, 1, 0)  # Convert to binary adjacency

    # Create a NetworkX DiGraph
    G = nx.from_numpy_array(W_binary, create_using=nx.DiGraph)

    # Assign weights (edge attributes) to existing edges
    for u, v in G.edges:
        G[u][v]['weight'] = 1

    return G


def generate_datasets_gp(node_counts, edge_probability, num_samples, noise_variance_range):
    """
    Generate datasets based on Gaussian Process (GP) causal relationships.

    Parameters:
        node_counts (list): List of node counts for each dataset.
        edge_probability (float): Probability of forming an edge.
        num_samples (int): Number of samples to generate.
        noise_variance_range (tuple): Range for sampling noise variance.

    Returns:
        dict: A dictionary of generated datasets for each node count.
        dict: A dictionary of adjacency matrices for each node count.
    """
    datasets = {}
    adjacency_matrices = {}

    for num_nodes in node_counts:
        print(f"\nGenerating data for {num_nodes} nodes...")

        # Generate a random DAG
        dag = generate_graph_with_edges(num_nodes, edge_probability)
        adjacency_matrix = nx.to_numpy_array(dag)
        adjacency_matrices[num_nodes] = adjacency_matrix

        # Initialize Gaussian Process (GP) functions
        gp_functions = {}
        kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
        for edge in dag.edges:
            gp_functions[edge] = GPy.models.GPRegression(np.zeros((1, 1)), np.zeros((1, 1)), kernel.copy())

        # Generate data
        data = np.zeros((num_samples, num_nodes))
        for node in nx.topological_sort(dag):
            parents = list(dag.predecessors(node))
            noise_variance = np.random.uniform(*noise_variance_range)

            for i in range(num_samples):
                if not parents:  # Root node
                    data[i, node] = np.random.normal(scale=np.sqrt(noise_variance))
                else:  # Nodes with parents
                    parent_values = data[i, parents].reshape(-1, 1)
                    gp_values = 0
                    for parent in parents:
                        gp = gp_functions[(parent, node)]
                        gp.set_XY(parent_values, parent_values)
                        gp_output = gp.posterior_samples_f(parent_values, size=1)
                        gp_values += gp_output.flatten()

                    data[i, node] = gp_values.sum() + np.random.normal(scale=np.sqrt(noise_variance))

        # Store generated data
        datasets[num_nodes] = data
        print(f"Data for {num_nodes} nodes generated with shape: {data.shape}")

        # Save adjacency matrix and data to CSV
        adj_matrix_df = pd.DataFrame(adjacency_matrix)
        adj_matrix_df.to_csv(f"adj_matrix_{num_nodes}nodes_GP.csv", index=False, header=[f"Feature {i}" for i in range(adj_matrix_df.shape[1])])

        data_df = pd.DataFrame(data)
        data_df.to_csv(f"gaussian_{num_nodes}nodes_GP.csv", index=False, header=[f"Feature {i}" for i in range(data_df.shape[1])])

    print("CSV files saved successfully.")
    return datasets, adjacency_matrices
