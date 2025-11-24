"""
MCI-based Fine-Grained Modeling Framework for Pipeline Latency Prediction
基于MCI的细粒度建模框架，用于预测pipeline在不同DOP下的延迟

架构：
Query Plan (Ch1) -> Plan Embedder (GTN) -> Plan Embedding -> Latency Predictor (MLP) -> Latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import numpy as np
import onnx
import onnxruntime as ort
import os
import time
import pandas as pd


class PlanEmbedder(nn.Module):
    """
    Plan Embedder using Graph Transformer Network (GTN)
    使用图转换器网络将查询计划嵌入为固定维度的向量
    """
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256, num_heads=8, num_layers=3):
        super(PlanEmbedder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection layer with BatchNorm
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Use GAT for better attention mechanism
            concat = True if i < num_layers - 1 else False
            
            self.conv_layers.append(
                GATConv(
                    hidden_dim, 
                    hidden_dim // num_heads, 
                    heads=num_heads,
                    dropout=0.1,
                    concat=concat
                )
            )
            # LayerNorm and BatchNorm dimension should match GAT output dimension
            norm_dim = hidden_dim if concat else hidden_dim // num_heads
            self.norm_layers.append(nn.LayerNorm(norm_dim))
            self.batch_norms.append(nn.BatchNorm1d(norm_dim))
        
        # Global pooling and final projection
        # Calculate final output dimension based on GAT concat setting
        # The last layer (i = num_layers - 1) has concat=False
        # When concat=False: output_dim = hidden_dim // num_heads
        # When concat=True: output_dim = (hidden_dim // num_heads) * num_heads = hidden_dim
        final_output_dim = hidden_dim // num_heads  # Last layer always has concat=False
            
        self.global_pool_proj = nn.Sequential(
            nn.Linear(final_output_dim * 2, hidden_dim),  # mean + max pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment for nodes (num_nodes,)
        
        Returns:
            plan_embedding: Plan embedding (batch_size, embedding_dim)
        """
        # Input projection
        h = self.input_projection(x)  # (num_nodes, hidden_dim)
        
        # Graph convolution with residual connections
        for i, (conv, norm, bn) in enumerate(zip(self.conv_layers, self.norm_layers, self.batch_norms)):
            h_residual = h
            h = conv(h, edge_index)
            h = bn(h)  # Apply BatchNorm
            h = norm(h)  # Apply LayerNorm
            h = F.relu(h)
            
            # Only use residual connection if dimensions match
            if h.size(1) == h_residual.size(1):
                h = h + h_residual  # Residual connection
        
        # Global pooling: mean + max
        # Use PyTorch Geometric pooling (works better with ONNX)
        h_mean = global_mean_pool(h, batch)  # (batch_size, final_output_dim)
        h_max = global_max_pool(h, batch)    # (batch_size, final_output_dim)
        h_pooled = torch.cat([h_mean, h_max], dim=1)  # (batch_size, final_output_dim * 2)
        
        # Global pooling projection
        h_pooled = self.global_pool_proj(h_pooled)  # (batch_size, hidden_dim)
        
        # Final embedding
        plan_embedding = self.embedding_proj(h_pooled)  # (batch_size, embedding_dim)
        
        return plan_embedding


class LatencyPredictor(nn.Module):
    """
    Latency Predictor using Multi-Layer Perceptron (MLP)
    使用多层感知机预测pipeline在指定DOP下的延迟
    """
    def __init__(self, embedding_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super(LatencyPredictor, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final prediction layer - single output for specified DOP
        self.mlp = nn.Sequential(*layers)
        self.predictor_head = nn.Linear(prev_dim, 1)
        
    def forward(self, plan_embedding):
        """
        Args:
            plan_embedding: Plan embedding (batch_size, embedding_dim)
        
        Returns:
            latency_predictions: Predicted latency (batch_size, 1)
        """
        # Pass through MLP
        features = self.mlp(plan_embedding)  # (batch_size, prev_dim)
        
        # Single prediction head
        predictions = self.predictor_head(features)  # (batch_size, 1)
        
        return predictions


class MCILatencyModel(nn.Module):
    """
    MCI-based Fine-Grained Latency Prediction Model
    基于MCI的细粒度延迟预测模型
    
    架构: Query Plan + DOP -> Plan Embedder (GTN) -> Plan Embedding + DOP -> Latency Predictor (MLP) -> Latency
    """
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256, 
                 num_heads=8, num_layers=3, predictor_hidden_dims=[512, 256, 128], 
                 dropout=0.2):
        super(MCILatencyModel, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Plan Embedder (GTN)
        self.plan_embedder = PlanEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Latency Predictor (MLP) - now includes DOP as input
        self.latency_predictor = LatencyPredictor(
            embedding_dim=embedding_dim + 1,  # +1 for DOP input
            hidden_dims=predictor_hidden_dims,
            dropout=dropout
        )
        
    def forward(self, x, edge_index, batch, dop_levels):
        """
        Args:
            x: Node features (num_nodes, input_dim) - 算子特征
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment for nodes (num_nodes,)
            dop_levels: DOP levels for each pipeline (batch_size,)
        
        Returns:
            latency_predictions: Predicted latency (batch_size, 1)
            plan_embeddings: Plan embeddings (batch_size, embedding_dim) - for analysis/debugging
        """
        # Plan embedding through GTN
        plan_embeddings = self.plan_embedder(x, edge_index, batch)  # (batch_size, embedding_dim)
        
        # Combine plan embedding with DOP level
        # Handle both scalar (0-d) and 1-d dop_levels
        if dop_levels.dim() == 0:
            # Scalar: expand to match batch size
            batch_size = plan_embeddings.size(0)
            dop_input = dop_levels.view(1, 1).expand(batch_size, 1).float()
        else:
            # 1-d or higher: ensure it's (batch_size, 1)
            dop_input = dop_levels.view(-1, 1).float()
        
        combined_features = torch.cat([plan_embeddings, dop_input], dim=1)  # (batch_size, embedding_dim + 1)
        
        # Latency prediction through MLP
        latency_predictions = self.latency_predictor(combined_features)
        
        return latency_predictions, plan_embeddings
    
    def get_plan_embeddings(self, x, edge_index, batch):
        """
        Get plan embeddings only (for analysis/debugging)
        """
        return self.plan_embedder(x, edge_index, batch)


def create_mci_model(input_dim, **kwargs):
    """
    Factory function to create MCI model with default parameters
    
    Args:
        input_dim: Input feature dimension for operators
        **kwargs: Additional model parameters
    
    Returns:
        MCILatencyModel instance
    """
    default_params = {
        'hidden_dim': 128,
        'embedding_dim': 256,
        'num_heads': 8,
        'num_layers': 3,
        'predictor_hidden_dims': [512, 256, 128],
        'dropout': 0.2
    }
    
    # Update default parameters with provided kwargs
    default_params.update(kwargs)
    
    return MCILatencyModel(input_dim=input_dim, **default_params)


def export_mci_model_to_onnx(model, output_path, input_dim, embedding_dim, device='cpu'):
    """
    Export MCI model to ONNX format for inference
    
    Args:
        model: Trained MCILatencyModel
        output_path: Path to save ONNX model
        input_dim: Input feature dimension
        embedding_dim: Embedding dimension
        device: Device to run on
    """
    model.eval()
    model.to(device)
    
    # Create dummy inputs for ONNX export
    batch_size = 2
    num_nodes = 5
    
    # Dummy node features
    x = torch.randn(num_nodes, input_dim, device=device)
    
    # Dummy edge index (fully connected)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long, device=device)
    
    # Dummy batch assignment
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    # Dummy DOP levels - ensure correct batch size
    dop_levels = torch.tensor([8, 16], dtype=torch.long, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (x, edge_index, batch, dop_levels),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['x', 'edge_index', 'batch', 'dop_levels'],
        output_names=['latency_predictions', 'plan_embeddings'],
        dynamic_axes={
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'batch': {0: 'num_nodes'},
            'dop_levels': {0: 'batch_size'},
            'latency_predictions': {0: 'batch_size'},
            'plan_embeddings': {0: 'batch_size'}
        }
    )
    
    print(f"MCI model exported to ONNX: {output_path}")


def predict_and_evaluate_mci_onnx(
    onnx_model_path,
    test_loader,
    output_dir,
    epsilon=1e-6
):
    """
    Use ONNX model for MCI pipeline latency prediction and evaluation
    
    Args:
        onnx_model_path: Path to ONNX model
        test_loader: PyG DataLoader for test data
        output_dir: Output directory for results
        epsilon: Small value to prevent division by zero
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    qerror_file = os.path.join(output_dir, "qerror.csv")
    comparison_file = os.path.join(output_dir, "comparison.csv")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    
    actual_times = []
    predicted_times = []
    pipeline_ids = []
    dop_levels = []
    inference_latencies = []
    
    total_start = time.time()
    
    for data in test_loader:
        # Prepare inputs for ONNX
        x_np = data.x.numpy().astype(np.float32)
        edge_index_np = data.edge_index.numpy().astype(np.int64)
        batch_np = data.batch.numpy().astype(np.int64)
        dop_np = data.dop_level.numpy().astype(np.float32)
        
        ort_inputs = {
            "x": x_np,
            "edge_index": edge_index_np,
            "batch": batch_np,
            "dop_levels": dop_np
        }
        
        # Run inference and measure latency
        start_time = time.time()
        outputs = session.run(["latency_predictions"], ort_inputs)
        end_time = time.time()
        
        inference_latencies.append(end_time - start_time)
        
        # Get predictions
        pred_time = outputs[0].flatten()
        true_time = data.y.numpy().flatten()
        
        # Store results
        actual_times.extend(true_time)
        predicted_times.extend(pred_time)
        
        # Get pipeline metadata
        if hasattr(data, 'pipeline_metadata'):
            for i in range(len(true_time)):
                pipeline_ids.append(f"pipeline_{data.thread_id[i].item()}")
        else:
            pipeline_ids.extend([f"pipeline_{i}" for i in range(len(true_time))])
        
        dop_levels.extend(dop_np)
    
    total_duration = time.time() - total_start
    
    # Convert to numpy arrays
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    dop_levels = np.array(dop_levels)
    
    # Calculate Q-error
    q_error = np.maximum(actual_times / (predicted_times + epsilon), 
                        predicted_times / (actual_times + epsilon)) - 1
    
    # Calculate other metrics
    mse = np.mean((predicted_times - actual_times) ** 2)
    mae = np.mean(np.abs(predicted_times - actual_times))
    mape = np.mean(np.abs((predicted_times - actual_times) / (actual_times + epsilon))) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "pipeline_id": pipeline_ids,
        "dop": dop_levels.astype(int),
        "actual_time": actual_times,
        "predicted_time": predicted_times,
        "q_error": q_error
    })
    
    # Save comparison results
    results_df.to_csv(comparison_file, index=False)
    
    # Calculate Q-error statistics
    qerror_stats = {
        "mean_q_error": np.mean(q_error),
        "median_q_error": np.median(q_error),
        "max_q_error": np.max(q_error),
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "num_samples": len(actual_times),
        "avg_inference_latency": np.mean(inference_latencies),
        "total_inference_time": total_duration
    }
    
    # Save Q-error statistics
    qerror_df = pd.DataFrame([qerror_stats])
    qerror_df.to_csv(qerror_file, index=False)
    
    print(f"ONNX inference completed:")
    print(f"  Mean Q-error: {qerror_stats['mean_q_error']:.4f}")
    print(f"  Median Q-error: {qerror_stats['median_q_error']:.4f}")
    print(f"  MSE: {qerror_stats['mse']:.4f}")
    print(f"  MAE: {qerror_stats['mae']:.4f}")
    print(f"  Average inference latency: {qerror_stats['avg_inference_latency']:.6f}s")
    
    return qerror_stats


# Loss function for MCI model
def mci_loss(predictions, targets, loss_type='mse', alpha=0.1):
    """
    Loss function for MCI model
    
    Args:
        predictions: Model predictions (batch_size, 1)
        targets: Ground truth latency values (batch_size,)
        loss_type: Type of loss function ('mse', 'mae', 'huber')
        alpha: Weight for relative error component
    
    Returns:
        loss: Combined loss value
    """
    # Single DOP prediction - direct comparison
    selected_predictions = predictions.squeeze()
    
    # Basic loss
    if loss_type == 'mse':
        basic_loss = F.mse_loss(selected_predictions, targets)
    elif loss_type == 'mae':
        basic_loss = F.l1_loss(selected_predictions, targets)
    elif loss_type == 'huber':
        basic_loss = F.huber_loss(selected_predictions, targets)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Relative error component for better training stability
    relative_error = torch.abs(selected_predictions - targets) / (targets + 1e-8)
    relative_loss = torch.mean(relative_error)
    
    # Combined loss
    total_loss = basic_loss + alpha * relative_loss
    
    return total_loss

