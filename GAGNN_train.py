import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
# Note: Ensure the 'data' folder is in the same directory as this script
adjacency_matrix = pd.read_csv("./data/adjacency_matrix.csv", index_col=0).values
semantic_adjacency_matrix = pd.read_csv("./data/semantic_adjacency_matrix_bus.csv", index_col=0).values
poi_features = pd.read_csv("./data/poi_count_by_region.csv", index_col=0).values
commute_flow_matrix = pd.read_csv("./data/commute_flow_matrix.csv", index_col=0).values
region_distance_matrix = pd.read_csv("./data/commute_flow_matrix45.csv", index_col=0).values

# Data preprocessing
poi_features = poi_features / np.max(poi_features, axis=0)  # Normalize POI features
region_distance_matrix = region_distance_matrix / np.max(region_distance_matrix)  # Normalize distance matrix
labels = commute_flow_matrix.flatten()  # Flatten commute flow matrix
distance_features = region_distance_matrix.flatten()[:, None]  # Inter-region distance features

# Construct region pairs
num_regions = adjacency_matrix.shape[0]
region_pairs = np.array([(i, j) for i in range(num_regions) for j in range(num_regions)])

# Convert to PyTorch tensors
adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
semantic_adjacency_matrix_tensor = torch.tensor(semantic_adjacency_matrix, dtype=torch.float32)
poi_features_tensor = torch.tensor(poi_features, dtype=torch.float32)
region_pairs_tensor = torch.tensor(region_pairs, dtype=torch.long)
distance_features_tensor = torch.tensor(distance_features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Split data into training and validation sets
train_indices, val_indices = train_test_split(
    np.arange(len(region_pairs_tensor)),
    test_size=0.3,
    random_state=42  # Fix random seed
)

train_dataset = TensorDataset(
    region_pairs_tensor[train_indices],
    distance_features_tensor[train_indices],
    labels_tensor[train_indices]
)
val_dataset = TensorDataset(
    region_pairs_tensor[val_indices],
    distance_features_tensor[val_indices],
    labels_tensor[val_indices]
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define single GAT layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, geo_adj, sem_adj, features):
        h = self.W(features)  # Linear transformation
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)  # Concatenate attention input
        e = self.leakyrelu(self.a(a_input).squeeze(1).view(N, N))  # Attention scores

        # Combine adjacency matrices
        combined_adj = geo_adj + sem_adj

        # Multiply attention scores with corresponding values in combined adjacency matrix
        e = e * combined_adj  # Element-wise multiplication
        
        zero_vec = -9e15 * torch.ones_like(e)  # Mask invalid connections
        attention = torch.where(combined_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # Softmax normalization
        return F.elu(torch.matmul(attention, h))  # GAT output

# Define MLP module
class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        return self.fc2(x)

# Define complete model
class GAT_Model(nn.Module):
    def __init__(self, gat_in, gat_out, mlp_input, mlp_hidden, mlp_output, num_gat_layers=2):
        super(GAT_Model, self).__init__()
        
        # Use multiple GATLayers to increase depth
        self.gat_layers = nn.ModuleList([GATLayer(gat_in if i == 0 else gat_out, gat_out) for i in range(num_gat_layers)])
        
        # MLP module
        self.mlp_module = MLPModule(mlp_input, mlp_hidden, mlp_output)

    def forward(self, geo_adj, sem_adj, features, region_pairs, distance_features):
        # Compute node embeddings through multiple GAT layers
        encoded_features = features
        for gat_layer in self.gat_layers:
            encoded_features = gat_layer(geo_adj, sem_adj, encoded_features)

        # Extract features for each region pair
        region1_features = encoded_features[region_pairs[:, 0]]  # Origin region features
        region2_features = encoded_features[region_pairs[:, 1]]  # Destination region features

        # Concatenate region features and distance features
        combined_features = torch.cat([region1_features, region2_features, distance_features], dim=1)

        # Predict via MLP
        return self.mlp_module(combined_features)

# Initialize model, set number of GAT layers
gat_in = poi_features.shape[1]  # Input feature dimension (POI feature dimension)
gat_out = 8  # GAT output feature dimension
mlp_input = 2 * gat_out + 1  # MLP input features (two region features + distance feature)
mlp_hidden = 128
mlp_output = 1  # Output commuting flow

num_gat_layers = 3  # Set hidden GAT layers to 3

model = GAT_Model(gat_in, gat_out, mlp_input, mlp_hidden, mlp_output, num_gat_layers=num_gat_layers)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training and validation
loss_history = []
val_loss_history = []
epochs = 700
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.001
    for batch_region_pairs, batch_distance_features, batch_labels in train_loader:
        predictions = model(
            adjacency_matrix_tensor, semantic_adjacency_matrix_tensor,
            poi_features_tensor, batch_region_pairs, batch_distance_features
        )
        batch_labels = batch_labels.view(-1, 1)
        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    loss_history.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_region_pairs, batch_distance_features, batch_labels in val_loader:
            predictions = model(
                adjacency_matrix_tensor, semantic_adjacency_matrix_tensor,
                poi_features_tensor, batch_region_pairs, batch_distance_features
            )
            batch_labels = batch_labels.view(-1, 1)
            loss = loss_function(predictions, batch_labels)
            val_loss += loss.item()
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# Visualize training and validation loss curves
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()