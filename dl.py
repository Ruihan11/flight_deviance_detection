"""
Enhanced Deep Learning Framework with Comprehensive Monitoring
- Modular architecture with dependency injection
- Robust error handling and graceful degradation  
- Real-time monitoring and observability
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# AMP (兼容导入)
try:
    from torch.amp.autocast_mode import autocast
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
        print("⚠️ AMP not available, falling back to FP32")

if AMP_AVAILABLE:
    from torch.cuda.amp import GradScaler

# CUDA Tools with fallback
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    # Mock NVTX for compatibility
    class MockNVTX:
        @staticmethod
        def range_push(msg): pass
        @staticmethod
        def range_pop(): pass
    nvtx = MockNVTX()

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ================== Enhanced Configuration ==================
@dataclass
class Config:
    """Centralized configuration with validation"""
    # Data
    n_samples: int = 10000
    n_features: int = 20
    n_classes: int = 3
    test_size: float = 0.2
    
    # Model
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    
    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 10
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    
    # Acceleration
    device: str = "auto"  # auto, cuda, cpu
    use_mixed_precision: bool = True
    use_cuda_graphs: bool = True
    compile_model: bool = True
    use_cudnn_benchmark: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_interval: float = 1.0  # seconds
    profile_memory: bool = True
    track_gradients: bool = True
    
    # Output
    results_dir: str = "results"
    experiment_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.warning("⚠️ CUDA not available, using CPU")
        
        # Create directory structure
        self.fig_dir = os.path.join(self.results_dir, self.experiment_name, "figs")
        self.table_dir = os.path.join(self.results_dir, self.experiment_name, "tables")
        self.model_dir = os.path.join(self.results_dir, self.experiment_name, "models")
        self.log_dir = os.path.join(self.results_dir, self.experiment_name, "logs")
        self.monitor_dir = os.path.join(self.results_dir, self.experiment_name, "monitoring")
        
        for dir_path in [self.fig_dir, self.table_dir, self.model_dir, self.log_dir, self.monitor_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Validate and adjust configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate and auto-adjust configuration based on environment"""
        # Check AMP availability
        if self.use_mixed_precision and not AMP_AVAILABLE:
            logger.warning("⚠️ AMP requested but not available, disabling")
            self.use_mixed_precision = False
        
        # Check CUDA for specific features
        if self.device == "cpu":
            if self.use_cuda_graphs:
                logger.warning("⚠️ CUDA Graphs not available on CPU")
                self.use_cuda_graphs = False
            if self.use_mixed_precision:
                logger.warning("⚠️ Mixed precision not beneficial on CPU")
                self.use_mixed_precision = False
        
        # Check torch.compile availability
        if self.compile_model and not hasattr(torch, 'compile'):
            logger.warning("⚠️ torch.compile not available (PyTorch < 2.0)")
            self.compile_model = False
        
        # Mutual exclusivity
        if self.compile_model and self.use_cuda_graphs:
            logger.info("📝 torch.compile takes precedence over CUDA Graphs")
            self.use_cuda_graphs = False


# ================== Monitoring Infrastructure ==================
class MetricsTracker:
    """Thread-safe metrics collection and aggregation"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
        self.lock = threading.Lock()
        
    def update(self, key: str, value: float, timestamp: Optional[float] = None):
        """Update a metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((timestamp, value))
            self.metrics[key] = value
    
    def get(self, key: str) -> Optional[float]:
        """Get current metric value"""
        with self.lock:
            return self.metrics.get(key)
    
    def get_history(self, key: str) -> List[Tuple[float, float]]:
        """Get metric history"""
        with self.lock:
            return self.history.get(key, []).copy()
    
    def get_all_current(self) -> Dict[str, float]:
        """Get all current metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def save_to_json(self, filepath: str):
        """Save metrics to JSON file"""
        with self.lock:
            data = {
                "current": self.metrics,
                "history": {k: v[-100:] for k, v in self.history.items()}  # Last 100 points
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class SystemMonitor(threading.Thread):
    """Real-time system monitoring thread"""
    
    def __init__(self, config: Config, metrics_tracker: MetricsTracker):
        super().__init__(daemon=True)
        self.config = config
        self.metrics = metrics_tracker
        self.running = False
        self.gpu_handle = None
        
        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("✅ GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"⚠️ GPU monitoring unavailable: {e}")
    
    def run(self):
        """Monitor system metrics in background"""
        self.running = True
        while self.running:
            try:
                # CPU metrics
                self.metrics.update("cpu_percent", self._get_cpu_usage())
                
                # Memory metrics
                mem_info = self._get_memory_info()
                for key, value in mem_info.items():
                    self.metrics.update(f"memory_{key}", value)
                
                # GPU metrics
                if self.gpu_handle:
                    gpu_info = self._get_gpu_info()
                    for key, value in gpu_info.items():
                        self.metrics.update(f"gpu_{key}", value)
                
                # PyTorch CUDA metrics
                if torch.cuda.is_available():
                    cuda_info = self._get_cuda_memory_info()
                    for key, value in cuda_info.items():
                        self.metrics.update(f"cuda_{key}", value)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.config.monitor_interval)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory info"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / 1024**3,
                "used_gb": mem.used / 1024**3,
                "percent": mem.percent
            }
        except ImportError:
            return {}
    
    def _get_gpu_info(self) -> Dict[str, float]:
        """Get GPU metrics from NVML"""
        if not self.gpu_handle:
            return {}
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                "memory_used_gb": mem_info.used / 1024**3,
                "memory_total_gb": mem_info.total / 1024**3,
                "memory_percent": (mem_info.used / mem_info.total) * 100,
                "temperature": pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0),
                "power_watts": pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000,
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu,
            }
        except Exception:
            return {}
    
    def _get_cuda_memory_info(self) -> Dict[str, float]:
        """Get PyTorch CUDA memory stats"""
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "active_gb": torch.cuda.memory_allocated() / 1024**3,
        }


class TrainingMonitor:
    """Monitor training-specific metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsTracker()
        self.system_monitor = None
        
        if config.enable_monitoring:
            self.system_monitor = SystemMonitor(config, self.metrics)
            self.system_monitor.start()
            logger.info("🔍 Training monitor started")
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float, val_acc: float, lr: float, epoch_time: float):
        """Log epoch metrics"""
        self.metrics.update("epoch", epoch)
        self.metrics.update("train_loss", train_loss)
        self.metrics.update("train_accuracy", train_acc)
        self.metrics.update("val_loss", val_loss)
        self.metrics.update("val_accuracy", val_acc)
        self.metrics.update("learning_rate", lr)
        self.metrics.update("epoch_time", epoch_time)
        
        # Calculate rates
        if epoch > 0:
            samples_per_sec = (self.config.batch_size * 
                             (self.config.n_samples * (1 - self.config.test_size)) / 
                             self.config.batch_size) / epoch_time
            self.metrics.update("training_throughput", samples_per_sec)
    
    def log_batch(self, batch_idx: int, loss: float, batch_time: float):
        """Log batch-level metrics"""
        self.metrics.update("batch_loss", loss)
        self.metrics.update("batch_time", batch_time)
        self.metrics.update("batch_throughput", self.config.batch_size / batch_time)
    
    def log_gradients(self, model: nn.Module):
        """Log gradient statistics"""
        if not self.config.track_gradients:
            return
        
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            self.metrics.update("gradient_norm_mean", np.mean(grad_norms))
            self.metrics.update("gradient_norm_max", np.max(grad_norms))
            self.metrics.update("gradient_norm_min", np.min(grad_norms))
    
    def save_monitoring_data(self):
        """Save all monitoring data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.monitor_dir, f"metrics_{timestamp}.json")
        self.metrics.save_to_json(metrics_file)
        
        # Generate monitoring report
        self._generate_report()
        
        logger.info(f"📊 Monitoring data saved to {self.config.monitor_dir}")
    
    def _generate_report(self):
        """Generate HTML monitoring report"""
        metrics = self.metrics.get_all_current()
        
        html_content = f"""
        <html>
        <head>
            <title>Training Monitor Report - {self.config.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px;
                    background: #f9f9f9;
                }}
                .metric-name {{ font-weight: bold; color: #333; }}
                .metric-value {{ font-size: 1.2em; color: #007bff; }}
                h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
            </style>
        </head>
        <body>
            <h1>🔍 Training Monitor Report</h1>
            <h2>Experiment: {self.config.experiment_name}</h2>
            
            <h3>📈 Training Metrics</h3>
            <div class="metrics-container">
        """
        
        # Add metrics
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            html_content += f"""
                <div class="metric">
                    <div class="metric-name">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        report_file = os.path.join(self.config.monitor_dir, "monitoring_report.html")
        with open(report_file, 'w') as f:
            f.write(html_content)
    
    def stop(self):
        """Stop monitoring"""
        if self.system_monitor:
            self.system_monitor.stop()
            self.save_monitoring_data()


# ================== Enhanced Data Pipeline ==================
class IDataPreparer(ABC):
    """Abstract interface for data preparation"""
    
    @abstractmethod
    def prepare_data(self) -> Tuple:
        pass
    
    @abstractmethod
    def create_dataloaders(self, *args, **kwargs) -> Tuple:
        pass


class DataPreparer(IDataPreparer):
    """Enhanced data preparation with monitoring"""
    
    def __init__(self, config: Config, monitor: Optional[TrainingMonitor] = None):
        self.config = config
        self.monitor = monitor
        self.scaler = None
        
    def prepare_data(self) -> Tuple:
        """Generate and prepare dataset with monitoring"""
        logger.info("📊 Generating dataset...")
        start_time = time.time()
        
        try:
            X, y = make_classification(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                n_informative=15,
                n_redundant=5,
                n_classes=self.config.n_classes,
                random_state=42,
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=42
            )
            
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            prep_time = time.time() - start_time
            
            # Log metrics
            if self.monitor:
                self.monitor.metrics.update("data_prep_time", prep_time)
                self.monitor.metrics.update("train_samples", len(X_train))
                self.monitor.metrics.update("test_samples", len(X_test))
            
            logger.info(f"✅ Data prepared in {prep_time:.2f}s")
            logger.info(f"   Training: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, self.scaler
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test) -> Tuple[DataLoader, DataLoader]:
        """Create optimized PyTorch dataloaders"""
        device = torch.device(self.config.device)
        
        try:
            # Direct GPU tensors for small datasets
            X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
            y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
            X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            y_test_t = torch.as_tensor(y_test, dtype=torch.long, device=device)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            test_dataset = TensorDataset(X_test_t, y_test_t)
            
            # Optimized DataLoader settings
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,  # For CUDA Graphs
                pin_memory=False,  # Already on GPU
                num_workers=0,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=0,
            )
            
            logger.info(f"✅ DataLoaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"❌ DataLoader creation failed: {e}")
            # Fallback to CPU
            if device.type == 'cuda':
                logger.warning("⚠️ Falling back to CPU tensors")
                self.config.device = 'cpu'
                return self.create_dataloaders(X_train, X_test, y_train, y_test)
            raise


# ================== Enhanced Model Architecture ==================
class BaseOptimizedModel(nn.Module):
    """Base class with common optimization features"""
    
    def __init__(self):
        super().__init__()
        self.training_step = 0
        
    def get_num_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> float:
        """Estimate model memory in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024**2


class OptimizedDeepNeuralNetwork(BaseOptimizedModel):
    """Enhanced MLP with monitoring hooks"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()
        
        logger.info(f"✅ Model created: {self.get_num_parameters():,} parameters, "
                   f"{self.get_memory_footprint():.2f} MB")
    
    def _init_weights(self):
        """Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)


# ================== Enhanced Trainer with Full Monitoring ==================
class OptimizedTrainer:
    """Production-grade trainer with comprehensive monitoring"""
    
    def __init__(self, model: nn.Module, config: Config, monitor: TrainingMonitor):
        self.config = config
        self.monitor = monitor
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        
        # Setup optimizations
        self._setup_optimization()
        self._setup_optimizer()
        self._setup_cuda_graphs()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
    def _setup_optimization(self):
        """Configure all optimization strategies"""
        self.compiled = False
        
        # torch.compile
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                logger.info("🔧 Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.compiled = True
                self.monitor.metrics.update("model_compiled", 1.0)
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")
                self.monitor.metrics.update("model_compiled", 0.0)
        
        # Mixed Precision
        self.use_amp = self.config.use_mixed_precision and AMP_AVAILABLE
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
        
        # CuDNN
        if self.config.use_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("✅ CuDNN benchmark mode enabled")
    
    def _setup_optimizer(self):
        """Setup optimizer with fallback"""
        try:
            # Try fused optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01,
                fused=True
            )
            logger.info("✅ Using fused AdamW optimizer")
        except (TypeError, RuntimeError):
            # Fallback to standard optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            logger.info("📝 Using standard AdamW optimizer")
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_cuda_graphs(self):
        """Setup CUDA Graphs for inference"""
        self.graph = None
        self.static_input = None
        self.static_output = None
        
        if (self.config.use_cuda_graphs and 
            torch.cuda.is_available() and 
            not self.compiled):
            logger.info("🚀 CUDA Graphs will be initialized for inference")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with comprehensive monitoring"""
        logger.info("\n" + "="*60)
        logger.info("🎯 Starting Optimized Training")
        logger.info("="*60)
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos"
        )
        
        # Training loop
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, epoch, scheduler)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            
            # Timing
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            lr = self.optimizer.param_groups[0]['lr']
            self.monitor.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time)
            
            # Log to console
            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"LR: {lr:.6f} | Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint if best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr,
                'epoch_time': epoch_time
            })
        
        # Save final results
        self._save_training_results(training_history)
        
        return training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, scheduler):
        """Train one epoch with monitoring"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # NVTX profiling for selected epochs
        if NVTX_AVAILABLE and epoch % 10 == 0:
            nvtx.range_push(f"train_epoch_{epoch}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()
            
            # Zero gradients
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast(device_type='cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()
                    
                    # Log gradients
                    if self.config.track_gradients:
                        self.monitor.log_gradients(self.model)
            else:
                # Standard training
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    scheduler.step()
            
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Batch monitoring
            batch_time = time.time() - batch_start
            if batch_idx % 10 == 0:
                self.monitor.log_batch(batch_idx, loss.item(), batch_time)
            
            self.global_step += 1
        
        if NVTX_AVAILABLE and epoch % 10 == 0:
            nvtx.range_pop()
        
        return total_loss / len(train_loader), 100.0 * correct / total
    
    def _validate(self, val_loader: DataLoader):
        """Validation with optional CUDA Graphs"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Setup CUDA Graph for first validation
        if (self.config.use_cuda_graphs and 
            self.graph is None and 
            torch.cuda.is_available() and 
            not self.compiled):
            self._setup_cuda_graph_inference(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                # Try CUDA Graph if available
                if self.graph and data.shape[0] == self.static_input.shape[0]:
                    self.static_input.copy_(data)
                    self.graph.replay()
                    output = self.static_output
                else:
                    if self.use_amp:
                        with autocast(device_type='cuda'):
                            output = self.model(data)
                    else:
                        output = self.model(data)
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(val_loader), 100.0 * correct / total
    
    def _setup_cuda_graph_inference(self, val_loader: DataLoader):
        """Initialize CUDA Graph for inference"""
        try:
            sample_data, _ = next(iter(val_loader))
            self.static_input = torch.zeros_like(sample_data)
            
            # Warmup
            for _ in range(3):
                _ = self.model(self.static_input)
            
            torch.cuda.synchronize()
            
            # Capture graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.model(self.static_input)
            
            logger.info("✅ CUDA Graph captured for inference")
        except Exception as e:
            logger.warning(f"⚠️ CUDA Graph setup failed: {e}")
            self.graph = None
    
    def _save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': asdict(self.config)
        }
        
        checkpoint_path = os.path.join(
            self.config.model_dir, 
            f'checkpoint_epoch_{epoch}_acc_{val_acc:.2f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best model
        best_path = os.path.join(self.config.model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_training_results(self, history: List[Dict]):
        """Save comprehensive training results"""
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Save as CSV
        csv_path = os.path.join(self.config.table_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as JSON with metadata
        results = {
            'config': asdict(self.config),
            'history': history,
            'best_val_acc': self.best_val_acc,
            'total_steps': self.global_step,
            'model_params': self.model.get_num_parameters(),
            'model_memory_mb': self.model.get_memory_footprint()
        }
        
        json_path = os.path.join(self.config.table_dir, 'training_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Results saved to {self.config.table_dir}")


# ================== Visualization Module ==================
class Visualizer:
    """Generate comprehensive visualizations"""
    
    def __init__(self, config: Config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_history(self, history: List[Dict]):
        """Plot training curves"""
        df = pd.DataFrame(history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(df['epoch'], df['lr'], color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time
        axes[1, 1].bar(df['epoch'], df['epoch_time'], color='steelblue', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.config.fig_dir, 'training_history.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training history plot saved to {fig_path}")
    
    def plot_system_metrics(self, monitor: TrainingMonitor):
        """Plot system monitoring metrics"""
        metrics_to_plot = [
            ('gpu_utilization', 'GPU Utilization (%)', 'green'),
            ('gpu_memory_percent', 'GPU Memory (%)', 'blue'),
            ('gpu_temperature', 'GPU Temperature (°C)', 'red'),
            ('gpu_power_watts', 'GPU Power (W)', 'orange')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric_key, label, color) in enumerate(metrics_to_plot):
            history = monitor.metrics.get_history(metric_key)
            if history:
                timestamps, values = zip(*history)
                # Convert to relative time
                start_time = timestamps[0]
                rel_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
                
                axes[idx].plot(rel_times, values, color=color, linewidth=2, alpha=0.7)
                axes[idx].fill_between(rel_times, values, alpha=0.3, color=color)
                axes[idx].set_xlabel('Time (minutes)')
                axes[idx].set_ylabel(label)
                axes[idx].set_title(label)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.config.fig_dir, 'system_metrics.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 System metrics plot saved to {fig_path}")


# ================== Main Orchestrator ==================
def main():
    """Main execution with full monitoring and error handling"""
    
    # Initialize configuration
    config = Config()
    
    # Setup monitoring
    monitor = TrainingMonitor(config)
    
    try:
        logger.info("🚀 Starting Enhanced Deep Learning Pipeline")
        logger.info(f"📁 Experiment: {config.experiment_name}")
        
        # Data preparation
        data_preparer = DataPreparer(config, monitor)
        X_train, X_test, y_train, y_test, scaler = data_preparer.prepare_data()
        train_loader, test_loader = data_preparer.create_dataloaders(
            X_train, X_test, y_train, y_test
        )
        
        # Model initialization
        model = OptimizedDeepNeuralNetwork(
            input_dim=config.n_features,
            hidden_dims=config.hidden_dims,
            output_dim=config.n_classes,
            dropout_rate=config.dropout_rate
        )
        
        # Training
        trainer = OptimizedTrainer(model, config, monitor)
        history = trainer.train(train_loader, test_loader)
        
        # Visualization
        visualizer = Visualizer(config)
        visualizer.plot_training_history(history)
        visualizer.plot_system_metrics(monitor)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            predictions = []
            targets = []
            for data, target in test_loader:
                output = model(data)
                predictions.extend(output.argmax(1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Generate classification report
        report = classification_report(targets, predictions, digits=4)
        report_path = os.path.join(config.table_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        final_acc = accuracy_score(targets, predictions) * 100
        
        logger.info("\n" + "="*60)
        logger.info("✅ Training Complete!")
        logger.info(f"🎯 Final Test Accuracy: {final_acc:.2f}%")
        logger.info(f"🏆 Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
        logger.info(f"📁 Results saved to: {config.results_dir}/{config.experiment_name}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        monitor.stop()
        logger.info("🔚 Pipeline shutdown complete")


if __name__ == "__main__":
    main()