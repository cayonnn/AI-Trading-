"""
LSTM Model Module
=================
Production-grade Bidirectional LSTM for price prediction and trend classification.

Features:
- Bidirectional LSTM with attention mechanism
- Multi-layer architecture with dropout
- Early stopping and learning rate scheduling
- Model checkpointing
- Both regression and classification support
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger


@dataclass
class LSTMConfig:
    """LSTM model configuration (v2.0)."""
    input_size: int
    hidden_size: int = 256  # UPGRADED: 128 → 256
    num_layers: int = 3     # UPGRADED: 2 → 3
    dropout: float = 0.3    # UPGRADED: 0.2 → 0.3
    bidirectional: bool = True
    use_attention: bool = True
    output_size: int = 1
    task: str = 'regression'  # 'regression' or 'classification'
    use_logits: bool = False  # If True, output raw logits (for BCEWithLogitsLoss)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention for LSTM outputs (v2.0).
    
    Features:
    - 4 attention heads
    - Better pattern capture
    - Improved context understanding
    """
    
    def __init__(self, hidden_size: int, bidirectional: bool = True, num_heads: int = 4):
        super().__init__()
        
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention to LSTM outputs.
        
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Self-attention
        attn_output, attention_weights = self.multihead_attn(
            lstm_output, lstm_output, lstm_output
        )
        
        # Residual connection + layer norm
        attn_output = self.layer_norm(lstm_output + attn_output)
        
        # Use mean pooling over sequence
        context_vector = attn_output.mean(dim=1)
        
        # Project
        context_vector = self.projection(context_vector)
        
        # Average attention weights across heads
        avg_weights = attention_weights.mean(dim=1)  # (batch, seq_len)
        
        return context_vector, avg_weights


# Keep backward compatibility alias
Attention = MultiHeadAttention


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with attention for time series prediction.
    
    Architecture:
    1. Bidirectional LSTM layers with dropout
    2. Attention mechanism (optional)
    3. Fully connected output layer
    
    Usage:
        config = LSTMConfig(input_size=50, hidden_size=128)
        model = LSTMModel(config)
        
        output = model(x)  # x: (batch, seq_len, features)
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Attention layer
        self.use_attention = config.use_attention
        if self.use_attention:
            self.attention = Attention(config.hidden_size, config.bidirectional)
        
        # Output layers
        lstm_output_size = config.hidden_size * self.num_directions
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size // 2, config.output_size)
        )
        
        # Final activation
        # When using BCEWithLogitsLoss, we need raw logits (no sigmoid)
        if config.use_logits:
            self.output_activation = nn.Identity()
        elif config.task == 'classification':
            self.output_activation = nn.Sigmoid() if config.output_size == 1 else nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"LSTMModel created: input={config.input_size}, hidden={config.hidden_size}, "
            f"layers={config.num_layers}, bidirectional={config.bidirectional}, "
            f"attention={config.use_attention}, task={config.task}"
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Model output (and attention weights if requested)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden states from both directions
            if self.bidirectional:
                context = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                context = h_n[-1]
            attention_weights = None
        
        # Output layer
        output = self.fc(context)
        output = self.output_activation(output)
        
        if return_attention and attention_weights is not None:
            return output, attention_weights
        
        return output


class LSTMPredictor:
    """
    Wrapper class for training and prediction with LSTM model.
    
    Features:
    - Automatic device selection (GPU/CPU)
    - Training with early stopping
    - Model checkpointing
    - Learning rate scheduling
    - Class weight support for imbalanced data
    
    Usage:
        predictor = LSTMPredictor(
            input_size=50,
            hidden_size=128,
            task='classification'
        )
        
        predictor.fit(X_train, y_train, X_val, y_val)
        predictions = predictor.predict(X_test)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,       # UPGRADED: 128 → 256
        num_layers: int = 3,          # UPGRADED: 2 → 3
        dropout: float = 0.3,         # UPGRADED: 0.2 → 0.3
        bidirectional: bool = True,
        use_attention: bool = True,
        task: str = 'classification',
        learning_rate: float = 0.0003,  # UPGRADED: 0.0005 → 0.0003
        device: Optional[str] = None,
        class_weight: Optional[float] = None  # Weight for positive class
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            task: 'regression' or 'classification'
            learning_rate: Initial learning rate
            device: Device to use ('cuda' or 'cpu')
            class_weight: Weight for positive class (for imbalanced data)
        """
        # Determine if we should use logits (when using BCEWithLogitsLoss with class weights)
        use_logits_mode = class_weight is not None
        
        self.config = LSTMConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            output_size=1,
            task=task,
            use_logits=use_logits_mode  # Pass to model config
        )
        
        self.learning_rate = learning_rate
        self.task = task
        self.class_weight = class_weight
        
        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = LSTMModel(self.config).to(self.device)
        
        # Loss function - will be set up in fit() with class weights
        if task == 'classification':
            # Using BCEWithLogitsLoss for better numerical stability
            # Note: model output_activation will be set to Identity when using this
            if class_weight is not None:
                pos_weight = torch.tensor([class_weight]).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCELoss()
            self.use_logits = class_weight is not None
        else:
            self.criterion = nn.MSELoss()
            self.use_logits = False
        
        # Optimizer with increased weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # Increased from 1e-5
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Early stopping counter
        no_improve_count = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # For BCEWithLogitsLoss, we need raw logits
                if self.use_logits:
                    # Remove sigmoid from model output for loss calculation
                    loss = self.criterion(outputs, batch_y)
                    # Apply sigmoid for accuracy calculation
                    probs = torch.sigmoid(outputs)
                else:
                    loss = self.criterion(outputs, batch_y)
                    probs = outputs
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
                # Calculate accuracy for classification
                if self.task == 'classification':
                    predicted = (probs >= 0.5).float()
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            self.history['train_loss'].append(avg_train_loss)
            
            if self.task == 'classification':
                train_acc = train_correct / train_total
                self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                if self.task == 'classification':
                    self.history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_count = 0
                    
                    # Save checkpoint
                    if checkpoint_dir:
                        self.save(str(checkpoint_path / 'best_model.pt'))
                else:
                    no_improve_count += 1
                
                # Log progress
                if self.task == 'classification':
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )
                
                # Check early stopping
                if no_improve_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                
                if self.task == 'classification':
                    # Apply sigmoid if using logits
                    if self.use_logits:
                        probs = torch.sigmoid(outputs)
                    else:
                        probs = outputs
                    predicted = (probs >= 0.5).float()
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        return avg_val_loss, val_acc
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, seq_len, n_features)
            return_proba: Return probabilities instead of classes
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Apply sigmoid if using logits (BCEWithLogitsLoss)
            if getattr(self, 'use_logits', False):
                outputs = torch.sigmoid(outputs)
            
            outputs = outputs.cpu().numpy()
        
        if self.task == 'classification' and not return_proba:
            return (outputs >= 0.5).astype(int).flatten()
        
        return outputs.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.predict(X, return_proba=True)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Model loaded from {filepath}")
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.
        
        Args:
            X: Input features
            
        Returns:
            Attention weights (n_samples, seq_len)
        """
        if not self.config.use_attention:
            raise ValueError("Model does not use attention")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            _, attention_weights = self.model(X_tensor, return_attention=True)
        
        return attention_weights.cpu().numpy()


if __name__ == "__main__":
    # Test LSTM model
    print("=== Testing LSTM Model ===")
    
    # Create dummy data
    n_samples = 1000
    seq_len = 60
    n_features = 50
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create and train model
    predictor = LSTMPredictor(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        task='classification'
    )
    
    history = predictor.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32,
        patience=5
    )
    
    # Make predictions
    predictions = predictor.predict(X_test)
    accuracy = (predictions == y_test).mean()
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Predictions shape: {predictions.shape}")
