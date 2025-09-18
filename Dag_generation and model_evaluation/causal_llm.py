import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import LlamaModel, LlamaConfig
from tqdm import tqdm
import os
from transformers import GPT2Model, GPT2Config
from transformers import AutoModelForCausalLM,AutoConfig

class CausalDiscoveryLLM:
    def __init__(self, input_dim, output_dim, model_path=None):
        self.model = CausalDiscoveryModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        self.criterion = nn.BCELoss()
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def learn(self, data, num_epochs=10, batch_size=32, epsilon=0.1):
        """
        Train the model on the provided data.
        """
        env = GraphEnvironment(data)

        for epoch in tqdm(range(num_epochs)):
            epoch_losses = []
            for _ in range(batch_size):
                state = env.get_next_state()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

                # Forward pass
                action_probs = self.model(state_tensor).squeeze(0).squeeze(0)
                action_probs = torch.sigmoid(action_probs)

                # Epsilon-greedy strategy
                if np.random.rand() < epsilon:
                    action_probs = torch.rand_like(action_probs)

                # Reshape to DAG format
                n = int(np.sqrt(action_probs.size(0)))
                action_probs_dag = action_probs.view(n, n)
                action_probs_dag = ensure_acyclic(action_probs_dag)

                # Flatten and compute loss
                action_probs_flat = action_probs_dag.view(-1)
                target = torch.zeros_like(action_probs_flat)
                loss = self.criterion(action_probs_flat, target)

                # Add L1 regularization
                l1_lambda = 0.01
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss += l1_lambda * l1_norm

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.requires_grad:
                        param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                # Store loss
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)





        # Save final model
        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)

    def causal_matrix(self, data):
        """
        Infer a causal adjacency matrix from the data.
        """
        data_tensor = torch.tensor(data, dtype=torch.float32)
        src = data_tensor.mean(dim=0, keepdim=True).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            adj_output = self.model(src).squeeze(0).squeeze(0)
            adj_probs = torch.sigmoid(adj_output).view(data.shape[1], data.shape[1])
            adj_probs = adj_probs * (1 - torch.eye(data.shape[1]))
            device = adj_probs.device
            print(adj_probs)
            adj_probs = graph_prunned_by_coef(adj_probs.detach().cpu().numpy(), data)
            adj_probs= torch.tensor(adj_probs, dtype=torch.float32, device=device)
            adj_matrix_final = ensure_acyclic(adj_probs)

        return adj_matrix_final.numpy()



class CausalDiscoveryModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CausalDiscoveryModel, self).__init__()

        # LLaMA configuration
        self.config = LlamaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            max_position_embeddings=512,
            vocab_size=32000  # Adjust based on your needs
        )

        # Initialize LLaMA model
        self.llama = LlamaModel(self.config) # Any LLM can be used (Llama,GPT,Gemma,DeepSeek)

        # Freeze all parameters of the LLaMA model
        for param in self.llama.parameters():
            param.requires_grad = False

        # Input projection
        self.input_projection = nn.Linear(input_dim, self.config.hidden_size)

        # Output projection
        self.output_projection = nn.Linear(self.config.hidden_size, output_dim)

    def forward(self, x):
        # Input projection
        x = x.to(torch.float32)  # Ensure input is float32
        x = self.input_projection(x)

        # Pass through LLaMA
        outputs = self.llama(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state

        # Output projection
        output = self.output_projection(hidden_states)

        return output


class GraphEnvironment:
    def __init__(self, synthetic_data):
        self.synthetic_data = synthetic_data
        self.current_state_index = 0
        self.current_state = None

    def get_next_state(self):
        # Move to the next sample in synthetic data
        self.current_state_index = (self.current_state_index + 1) % len(self.synthetic_data)
        self.current_state = self.synthetic_data[self.current_state_index]
        return self.current_state

    def reset(self):
        # Reset to the first sample in synthetic data
        self.current_state_index = 0
        self.current_state = self.synthetic_data[self.current_state_index]
        return self.current_state

def ensure_acyclic(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Takes a 2D PyTorch tensor 'adj_matrix' of shape (N, N).
    Interprets any adj_matrix[i,j] > 0 as an edge i->j with weight adj_matrix[i,j].
    Removes the lowest-weight edge in each cycle until no cycles remain.
    Returns an acyclic adjacency matrix (DAG).
    """
    if adj_matrix.dim() != 2:
        raise ValueError("ensure_acyclic expects a 2D adjacency matrix.")

    device = adj_matrix.device
    A = adj_matrix.detach().cpu().numpy()
    N = A.shape[0]

    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    while True:
        try:
            cycle_edges = nx.find_cycle(G, orientation='original')
            min_edge = None
            min_w = float('inf')
            for (u, v, direction) in cycle_edges:
                w_ = G[u][v]['weight']
                if w_ < min_w:
                    min_w = w_
                    min_edge = (u, v)

            G.remove_edge(min_edge[0], min_edge[1])
            A[min_edge[0], min_edge[1]] = 0.0

        except nx.NetworkXNoCycle:
            break

    return torch.from_numpy(A).float().to(device)


def calculate_threshold(adj_matrix, n):
    # Flatten the matrix and sort the weights in descending order
    flattened = adj_matrix.flatten()
    sorted_weights = np.sort(flattened)[::-1]

    # Get the n-th largest weight as the threshold
    threshold = sorted_weights[n-1]
    return threshold

def graph_prunned_by_coef(graph_batch, X):
    """
    For a given graph, prune edges according to edge weights from linear regression.
    graph_batch shape: [d, d], row i => child node i, columns => parent nodes (i <- parents).
    We'll keep edges where abs(coefficient) > th.
    """
    d = len(graph_batch)
    reg = LinearRegression()
    W = []

    for i in range(d):
        # Identify parents for node i
        # If your adjacency is strictly 0/1, you might do (graph_batch[i] == 1)
        col = np.abs(graph_batch[i]) > 0.5
        if np.sum(col) == 0:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        new_reg_coeff = np.zeros(d, dtype=float)
        parent_indices = np.where(col)[0]
        for idx_parent, coef_val in zip(parent_indices, reg_coeff):
            new_reg_coeff[idx_parent] = coef_val

        W.append(new_reg_coeff)

    W = np.array(W)
    # Prune edges whose |coef| <= th
    th =  calculate_threshold(W, X.shape[1])
    pruned = (np.abs(W) >= th).astype(np.float32)
    return pruned
