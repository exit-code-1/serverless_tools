"""
MCI Model Training Script for Pipeline Latency Prediction
MCI模型训练脚本，用于pipeline延迟预测

功能：
1. 训练MCI模型
2. 模型评估和验证
3. 模型保存和导出
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import time
import json
from typing import Dict, List, Tuple
import argparse

# Import our MCI modules
from mci_model import MCILatencyModel, create_mci_model, mci_loss, export_mci_model_to_onnx
from mci_data_loader import (
    load_mci_pipeline_data_exec,
    load_mci_pipeline_data_mem,
    create_mci_data_loaders_exec,
    create_mci_data_loaders_mem,
    analyze_data_statistics
)
from config import MCIConfig, create_config_file, PresetConfigs


class MCITrainer:
    """MCI Model Trainer"""
    
    def __init__(self, model: MCILatencyModel, config: MCIConfig, device: str = None):
        # Use device from config if not specified
        if device is None:
            if config.training.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = config.training.device
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.train_losses = []
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer: optim.Optimizer, 
                   loss_fn, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get predictions and embeddings
            predictions, embeddings = self.model(
                data.x, 
                data.edge_index, 
                data.batch,
                data.dop_level.squeeze()
            )
            
            # Compute loss
            targets = data.y.squeeze()
            
            loss = loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    
    def train(self, train_loader) -> Dict:
        """Train the model using configuration parameters"""
        
        # Setup optimizer and scheduler using config
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.training.learning_rate, 
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=max(1, self.config.training.epochs // 3),  # Ensure step_size >= 1
            gamma=0.5
        )
        
        # Setup logging
        writer = SummaryWriter(self.config.data.log_dir)
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training configuration:")
        print(f"  Epochs: {self.config.training.epochs}")
        print(f"  Learning rate: {self.config.training.learning_rate}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  DOP levels: {self.config.training.dop_levels}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, 
                                        lambda pred, target: mci_loss(
                                            pred, target, 
                                            self.config.training.loss_type, 
                                            self.config.training.loss_alpha
                                        ), epoch)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch:3d}/{self.config.training.epochs}: '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Time: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        
        writer.close()
        
        return {
            'final_train_loss': train_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'train_losses': self.train_losses
        }


def evaluate_model(model: MCILatencyModel, test_loader, device: str = 'cpu') -> Dict:
    """Evaluate model on test set"""
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    q_errors = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get predictions
            pred, _ = model(
                data.x, 
                data.edge_index, 
                data.batch,
                data.dop_level.squeeze()
            )
            
            target = data.y.squeeze()
            pred = pred.squeeze()
            
            # Calculate Q-error
            q_error = torch.max(pred / (target + 1e-8), target / (pred + 1e-8)) - 1
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            q_errors.extend(q_error.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    q_errors = np.array(q_errors)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    q_error_mean = np.mean(q_errors)
    q_error_median = np.median(q_errors)
    
    return {
        'mse': mse,
        'mae': mae,
        'q_error_mean': q_error_mean,
        'q_error_median': q_error_median,
        'predictions': predictions,
        'targets': targets
    }


def save_model_and_results(model: MCILatencyModel, results: Dict, 
                          output_dir: str, model_name: str = 'mci_model', config: MCIConfig = None):
    """Save model and training results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.plan_embedder.hidden_dim,
            'embedding_dim': model.plan_embedder.embedding_dim,
            'num_heads': model.plan_embedder.num_heads,
            'num_layers': model.plan_embedder.num_layers,
            'predictor_hidden_dims': config.model.predictor_hidden_dims if config else [256, 128],  # Use config or default
            'dropout': config.model.dropout if config else 0.1,
            'single_dop_prediction': True
        }
    }, model_path)
    
    # Save results
    results_path = os.path.join(output_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = value
            elif hasattr(value, '__class__') and 'MCILatencyModel' in str(value.__class__):
                # Skip model objects - they're saved separately as .pth files
                continue
            else:
                json_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        json.dump(json_results, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


def train_mci_execution_model(config: MCIConfig, device: str) -> Dict:
    """Train MCI execution time model"""
    print("=" * 60)
    print("Training MCI Execution Time Model")
    print("=" * 60)
    
    # Load training data for execution time
    train_csv_path = config.get_train_csv_path()
    train_info_csv_path = config.get_train_info_csv_path()
    
    print(f"Loading execution time training data...")
    print(f"Training CSV: {train_csv_path}")
    print(f"Training info CSV: {train_info_csv_path}")
    
    train_data = load_mci_pipeline_data_exec(
        csv_file=train_csv_path,
        query_info_file=train_info_csv_path,
        target_dop_levels=config.training.dop_levels,
        use_estimates=config.training.use_estimates
    )
    
    # Analyze data statistics
    train_stats = analyze_data_statistics(train_data, target_type='execution_time')
    print(f"Execution time training data statistics: {train_stats}")
    
    # Create training data loader
    train_loader = create_mci_data_loaders_exec(
        csv_file=train_csv_path,
        query_info_file=train_info_csv_path,
        target_dop_levels=config.training.dop_levels,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        use_estimates=config.training.use_estimates
    )[0]  # Only get train_loader
    
    # Create model
    print("Creating MCI execution time model...")
    model = create_mci_model(
        input_dim=train_stats['feature_dim'],
        hidden_dim=config.model.hidden_dim,
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        predictor_hidden_dims=config.model.predictor_hidden_dims,
        dropout=config.model.dropout
    )
    
    # Create trainer
    trainer = MCITrainer(model, config=config, device=device)
    
    # Train model
    print("Starting execution time model training...")
    training_results = trainer.train(train_loader=train_loader)
    
    # Skip ONNX export for now
    print("Skipping ONNX export - using PyTorch model only")
    
    return {
        'model': model,
        'training_results': training_results,
        'data_stats': train_stats
    }


def train_mci_memory_model(config: MCIConfig, device: str) -> Dict:
    """Train MCI memory model"""
    print("=" * 60)
    print("Training MCI Memory Model")
    print("=" * 60)
    
    # Load training data for memory usage
    train_csv_path = config.get_train_csv_path()
    train_info_csv_path = config.get_train_info_csv_path()
    
    print(f"Loading memory training data...")
    print(f"Training CSV: {train_csv_path}")
    print(f"Training info CSV: {train_info_csv_path}")
    
    train_data = load_mci_pipeline_data_mem(
        csv_file=train_csv_path,
        query_info_file=train_info_csv_path,
        target_dop_levels=config.training.dop_levels,
        use_estimates=config.training.use_estimates
    )
    
    # Analyze data statistics
    train_stats = analyze_data_statistics(train_data, target_type='memory_usage')
    print(f"Memory training data statistics: {train_stats}")
    
    # Create training data loader
    train_loader = create_mci_data_loaders_mem(
        csv_file=train_csv_path,
        query_info_file=train_info_csv_path,
        target_dop_levels=config.training.dop_levels,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        use_estimates=config.training.use_estimates
    )[0]  # Only get train_loader
    
    # Create model
    print("Creating MCI memory model...")
    model = create_mci_model(
        input_dim=train_stats['feature_dim'],
        hidden_dim=config.model.hidden_dim,
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        predictor_hidden_dims=config.model.predictor_hidden_dims,
        dropout=config.model.dropout
    )
    
    # Create trainer
    trainer = MCITrainer(model, config=config, device=device)
    
    # Train model
    print("Starting memory model training...")
    training_results = trainer.train(train_loader=train_loader)
    
    # Skip ONNX export for now
    print("Skipping ONNX export - using PyTorch model only")
    
    return {
        'model': model,
        'training_results': training_results,
        'data_stats': train_stats
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MCI Models for Pipeline Execution Time and Memory Prediction')
    
    # Only essential arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = MCIConfig.load_config(args.config)
    
    # Setup device
    if config.training.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.training.device
    
    print(f"Using device: {device}")
    print(f"Output directory: {config.data.output_dir}")
    
    # Record start time
    import time
    start_time = time.time()
    
    # Train execution time model
    print("\n" + "="*60)
    print("Starting Execution Time Model Training...")
    print("="*60)
    exec_start = time.time()
    exec_results = train_mci_execution_model(config, device)
    exec_time = time.time() - exec_start
    print(f"Execution time model training completed in {exec_time:.2f}s")
    
    # Train memory model
    print("\n" + "="*60)
    print("Starting Memory Model Training...")
    print("="*60)
    mem_start = time.time()
    mem_results = train_mci_memory_model(config, device)
    mem_time = time.time() - mem_start
    print(f"Memory model training completed in {mem_time:.2f}s")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results for both models
    exec_model_name = f"{config.data.model_name}_exec"
    mem_model_name = f"{config.data.model_name}_mem"
    
    save_model_and_results(exec_results['model'], exec_results, config.data.output_dir, exec_model_name, config)
    save_model_and_results(mem_results['model'], mem_results, config.data.output_dir, mem_model_name, config)
    
    # Create combined results summary
    combined_results = {
        'execution_model': {
            'training_results': exec_results['training_results'],
            'data_stats': exec_results['data_stats']
        },
        'memory_model': {
            'training_results': mem_results['training_results'],
            'data_stats': mem_results['data_stats']
        },
        'training_time': {
            'execution_model_time': exec_time,
            'memory_model_time': mem_time,
            'total_time': total_time
        },
        'model_config': config.to_dict(),
        'model_architecture': {
            'input_dim': exec_results['data_stats']['feature_dim'],
            'embedding_dim': config.model.embedding_dim,
            'single_dop_prediction': True,
            'dop_levels': config.training.dop_levels
        }
    }
    
    # Save combined results
    combined_results_path = os.path.join(config.data.output_dir, f"{config.data.model_name}_combined_results.json")
    with open(combined_results_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Execution time model saved as PyTorch model")
    print(f"Memory model saved as PyTorch model")
    print(f"Combined results saved to: {combined_results_path}")
    print()
    print("Training Time Summary:")
    print(f"  Execution model: {exec_time:.2f}s ({exec_time/60:.2f} min)")
    print(f"  Memory model: {mem_time:.2f}s ({mem_time/60:.2f} min)")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
