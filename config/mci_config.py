"""
MCI Model Configuration
MCI模型配置文件

This file contains all configurable parameters for MCI model training and inference.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """MCI Model Architecture Configuration"""
    # Model architecture parameters (optimized for CPU training)
    input_dim: int = 50                    # Input feature dimension (will be auto-detected)
    hidden_dim: int = 64                   # Hidden dimension for GNN layers (reduced for CPU)
    embedding_dim: int = 128               # Final embedding dimension (reduced for CPU)
    num_heads: int = 4                     # Number of attention heads in GAT (reduced for CPU)
    num_layers: int = 2                    # Number of GNN layers (reduced for CPU)
    predictor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # MLP hidden dimensions (reduced for CPU)
    dropout: float = 0.1                   # Dropout rate (reduced for smaller model)


@dataclass
class TrainingConfig:
    """Training Configuration (optimized for CPU)"""
    # Training parameters (optimized for CPU training)
    epochs: int = 80                       # Number of training epochs (reduced for CPU)
    batch_size: int = 32                   # Batch size for training (increased for CPU efficiency)
    learning_rate: float = 0.005           # Learning rate (slightly increased for faster convergence)
    weight_decay: float = 1e-4             # Weight decay for regularization
    
    # Loss function parameters
    loss_type: str = 'mae'                 # Loss type: 'mse', 'mae', 'huber'
    loss_alpha: float = 0.1                # Weight for relative error component
    
    # Data parameters
    dop_levels: Optional[List[int]] = None  # DOP levels to train on. If None, use all available DOPs
    use_estimates: bool = False            # Whether to use estimated values instead of actual values
    
    # System parameters (optimized for CPU)
    num_workers: int = 0                   # Number of worker processes for data loading (0 for CPU)
    pin_memory: bool = False               # Whether to pin memory (disabled for CPU)
    device: str = 'cpu'                    # Device to use: 'auto', 'cpu', 'cuda' (default to CPU)


@dataclass
class DataConfig:
    """Data Configuration"""
    # Dataset selection (统一使用项目配置)
    dataset: str = "tpch"                  # Dataset: 'tpch' or 'tpcds'
    
    # Output configuration
    output_dir: str = "./mci_output"       # Output directory for models and results
    model_name: str = "mci_model"          # Base name for saved models
    log_dir: str = "./mci_logs"            # Directory for training logs
    
    # Data processing
    max_nodes_per_graph: int = 1000        # Maximum nodes per graph (for memory management)
    min_pipeline_size: int = 1             # Minimum pipeline size to include
    max_pipeline_size: int = 50            # Maximum pipeline size to include


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    # Evaluation metrics
    epsilon_time: float = 1e-6             # Epsilon for time prediction stability
    epsilon_mem: float = 1e-2              # Epsilon for memory prediction stability
    
    # Q-error analysis
    q_error_bins: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    # Performance analysis
    analyze_by_dop: bool = True            # Whether to analyze performance by DOP level
    analyze_by_pipeline_type: bool = True  # Whether to analyze by pipeline type (streaming/non-streaming)


@dataclass
class MCIConfig:
    """Main MCI Configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Create output directories if they don't exist
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.data.log_dir, exist_ok=True)
    
    def get_train_csv_path(self) -> str:
        """Get training CSV file path"""
        from utils.dataset_loader import create_dataset_loader
        loader = create_dataset_loader(self.data.dataset)
        paths = loader.get_file_paths('train')
        return paths['plan_info']
    
    def get_train_info_csv_path(self) -> str:
        """Get training info CSV file path"""
        from utils.dataset_loader import create_dataset_loader
        loader = create_dataset_loader(self.data.dataset)
        paths = loader.get_file_paths('train')
        return paths['query_info']
    
    def get_test_csv_path(self) -> str:
        """Get test CSV file path"""
        from utils.dataset_loader import create_dataset_loader
        loader = create_dataset_loader(self.data.dataset)
        paths = loader.get_file_paths('test')
        return paths['plan_info']
    
    def get_test_info_csv_path(self) -> str:
        """Get test info CSV file path"""
        from utils.dataset_loader import create_dataset_loader
        loader = create_dataset_loader(self.data.dataset)
        paths = loader.get_file_paths('test')
        return paths['query_info']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def save_config(self, config_path: str):
        """Save configuration to JSON file"""
        import json
        
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to: {config_path}")
    
    @classmethod
    def load_config(cls, config_path: str) -> 'MCIConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            evaluation=evaluation_config
        )
    
        
        # Update training parameters if provided
        if hasattr(args, 'epochs') and args.epochs:
            self.training.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr:
            self.training.learning_rate = args.lr
        if hasattr(args, 'dop_levels') and args.dop_levels:
            self.training.dop_levels = args.dop_levels
        if hasattr(args, 'use_estimates') and args.use_estimates:
            self.training.use_estimates = args.use_estimates


# Default configuration instances
DEFAULT_CONFIG = MCIConfig()

# Predefined configurations for different scenarios
@dataclass
class PresetConfigs:
    """Predefined configurations for different use cases"""
    
    @staticmethod
    def small_model() -> MCIConfig:
        """Configuration for small model (optimized for CPU training)"""
        config = MCIConfig()
        config.model.hidden_dim = 32
        config.model.embedding_dim = 64
        config.model.num_heads = 2
        config.model.num_layers = 1
        config.model.predictor_hidden_dims = [128]
        config.training.batch_size = 64
        config.training.epochs = 40
        config.training.device = 'cpu'
        config.training.num_workers = 0
        config.training.pin_memory = False
        return config
    
    @staticmethod
    def large_model() -> MCIConfig:
        """Configuration for large model (better accuracy, optimized for GPU)"""
        config = MCIConfig()
        config.model.hidden_dim = 256
        config.model.embedding_dim = 512
        config.model.num_heads = 16
        config.model.num_layers = 4
        config.model.predictor_hidden_dims = [1024, 512, 256]
        config.training.batch_size = 8
        config.training.epochs = 200
        config.training.device = 'cuda'  # Optimized for GPU
        config.training.num_workers = 4
        config.training.pin_memory = True
        return config
    
    @staticmethod
    def quick_test() -> MCIConfig:
        """Configuration for quick testing (minimal CPU model)"""
        config = MCIConfig()
        config.model.hidden_dim = 16
        config.model.embedding_dim = 32
        config.model.num_heads = 1
        config.model.num_layers = 1
        config.model.predictor_hidden_dims = [64]
        config.training.batch_size = 128
        config.training.epochs = 5
        config.training.dop_levels = [1, 2]
        config.training.device = 'cpu'
        config.training.num_workers = 0
        config.training.pin_memory = False
        return config
    
    @staticmethod
    def production() -> MCIConfig:
        """Configuration for production use (optimized for CPU)"""
        config = MCIConfig()
        # Use default CPU-optimized settings but with more training
        config.training.epochs = 120
        config.training.learning_rate = 0.0015
        config.training.weight_decay = 1e-5
        config.training.dop_levels = [1, 2, 4, 8, 16, 32]
        config.data.max_nodes_per_graph = 2000
        config.training.device = 'cpu'
        config.training.num_workers = 0
        config.training.pin_memory = False
        return config


def create_config_file(config_path: str = "mci_config.json", preset: str = "default"):
    """Create a configuration file with specified preset"""
    if preset == "small":
        config = PresetConfigs.small_model()
    elif preset == "large":
        config = PresetConfigs.large_model()
    elif preset == "quick":
        config = PresetConfigs.quick_test()
    elif preset == "production":
        config = PresetConfigs.production()
    else:
        config = DEFAULT_CONFIG
    
    config.save_config(config_path)
    return config


if __name__ == "__main__":
    # Create example configuration files
    print("Creating example configuration files...")
    
    # Default configuration
    create_config_file("configs/default_config.json", "default")
    
    # Small model configuration
    create_config_file("configs/small_model_config.json", "small")
    
    # Large model configuration  
    create_config_file("configs/large_model_config.json", "large")
    
    # Quick test configuration
    create_config_file("configs/quick_test_config.json", "quick")
    
    # Production configuration
    create_config_file("configs/production_config.json", "production")
    
    print("✅ Configuration files created successfully!")
    print("\nAvailable presets:")
    print("- default: Standard configuration")
    print("- small: Fast training, lower accuracy")
    print("- large: Slow training, higher accuracy") 
    print("- quick: For quick testing")
    print("- production: For production deployment")
