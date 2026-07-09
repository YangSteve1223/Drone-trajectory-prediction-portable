"""Training logger and visualization: per-epoch metrics to JSON, loss curves, convergence check."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class TrainingLogger:
    """Training log manager."""

    def __init__(
        self,
        log_dir: str = './runs/default',
        patience: int = 15,           # convergence: suggest stop after N epochs without improvement
        min_delta: float = 0.001,     # min relative improvement threshold
        smoothing_window: int = 5,    # curve smoothing window
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / 'training_log.json'

        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window

        # Log records
        self.history: List[Dict] = []

        # Convergence tracking
        self.best_val_rmse = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Try to load existing log (resume)
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    self.history = json.load(f)
                if self.history:
                    self.best_val_rmse = min(
                        (e.get('val_rmse', float('inf')) for e in self.history),
                        default=float('inf')
                    )
                    self.best_epoch = next(
                        (e['epoch'] for e in self.history
                         if e.get('val_rmse', float('inf')) == self.best_val_rmse),
                        0
                    )
                    print(f"[Logger] Resumed from {len(self.history)} epochs, "
                          f"best RMSE={self.best_val_rmse:.4f} @ epoch {self.best_epoch}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Logger] Warning: Could not load existing log: {e}")
                self.history = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_breakdown: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """
        Log metrics for one epoch.

        Args:
            epoch: epoch number
            train_loss: total training loss
            train_breakdown: loss breakdown {'displacement': ..., 'intent': ..., ...}
            val_metrics: validation metrics (optional) {'RMSE': ..., 'Distance_Accuracy': ..., ...}
            lr: current learning rate (optional)
        """
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_loss_displacement': float(train_breakdown.get('displacement', 0)),
            'train_loss_intent': float(train_breakdown.get('intent', 0)),
            'train_loss_uncertainty': float(train_breakdown.get('uncertainty', 0)),
            'train_loss_physics': float(train_breakdown.get('physics', 0)),
        }
        if lr is not None:
            entry['lr'] = float(lr)

        if val_metrics:
            entry['val_rmse'] = float(val_metrics.get('RMSE', float('nan')))
            entry['val_mae'] = float(val_metrics.get('MAE', float('nan')))
            entry['val_dist_acc'] = float(val_metrics.get('Distance_Accuracy', float('nan')))
            entry['val_dir_acc'] = float(val_metrics.get('Direction_Accuracy', float('nan')))
            entry['val_intent_acc'] = float(val_metrics.get('intent_accuracy', float('nan')))
            entry['val_mean_jerk'] = float(val_metrics.get('Mean_Jerk', float('nan')))
            entry['val_loss'] = float(val_metrics.get('val_loss', float('nan')))

            # Update convergence tracking
            current_rmse = entry['val_rmse']
            # Handle initial state (best_val_rmse = inf)
            if self.best_val_rmse == float('inf') or np.isinf(self.best_val_rmse):
                improvement = 1.0  # first validation always an improvement
            else:
                improvement = (self.best_val_rmse - current_rmse) / max(self.best_val_rmse, 1e-8)
            if improvement > self.min_delta:
                self.best_val_rmse = current_rmse
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                entry['is_best'] = True
            else:
                self.epochs_without_improvement += 1
                entry['is_best'] = False

        self.history.append(entry)
        self._save_json()

    def _save_json(self):
        """Save log to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def check_convergence(self) -> Dict:
        """
        Check whether training has converged.

        Returns:
            dict with:
                converged: bool
                message: str
                best_epoch: int
                best_rmse: float
                suggestion: str
        """
        if len(self.history) < 10:
            return {
                'converged': False,
                'message': f'only {len(self.history)} epochs, need at least 10',
                'best_epoch': self.best_epoch if self.best_val_rmse != float('inf') else -1,
                'best_rmse': self.best_val_rmse,
                'suggestion': 'keep training'
            }

        # Only check training loss trend when no validation data
        has_val = any('val_rmse' in e for e in self.history)
        if not has_val:
            recent = self.history[-10:]
            losses = [e.get('train_loss', 0) for e in recent]
            if len(losses) >= 5:
                trend = (np.mean(losses[:5]) - np.mean(losses[-5:])) / max(np.mean(losses[:5]), 1e-8)
                if trend < 0.001:
                    return {
                        'converged': True,
                        'message': 'training loss has plateaued',
                        'best_epoch': -1,
                        'best_rmse': float('inf'),
                        'suggestion': 'add a validation set or keep observing'
                    }
            return {
                'converged': False,
                'message': 'no validation data, tracking train loss only',
                'best_epoch': -1,
                'best_rmse': float('inf'),
                'suggestion': 'keep training'
            }

        recent = self.history[-self.patience:]
        recent_losses = [e.get('train_loss', 0) for e in recent if 'train_loss' in e]

        if len(recent_losses) < self.patience // 2:
            return {
                'converged': False,
                'message': 'insufficient validation data',
                'best_epoch': self.best_epoch,
                'best_rmse': self.best_val_rmse,
                'suggestion': 'keep training for more validation points'
            }

        # Check training loss trend
        if len(recent_losses) >= 5:
            first_half = np.mean(recent_losses[:len(recent_losses) // 2])
            second_half = np.mean(recent_losses[len(recent_losses) // 2:])
            loss_trend = (first_half - second_half) / max(first_half, 1e-8)
        else:
            loss_trend = 0

        if self.epochs_without_improvement >= self.patience:
            return {
                'converged': True,
                'message': f'val RMSE has not improved for {self.epochs_without_improvement} epochs',
                'best_epoch': self.best_epoch,
                'best_rmse': self.best_val_rmse,
                'suggestion': f'stop training, use epoch {self.best_epoch} weights (RMSE={self.best_val_rmse:.4f})'
            }
        elif self.epochs_without_improvement >= self.patience * 0.6:
            return {
                'converged': False,
                'message': f'near convergence ({self.epochs_without_improvement}/{self.patience} epochs no improvement)',
                'best_epoch': self.best_epoch,
                'best_rmse': self.best_val_rmse,
                'suggestion': 'may converge soon, observe a few more epochs'
            }
        else:
            return {
                'converged': False,
                'message': 'still improving',
                'best_epoch': self.best_epoch,
                'best_rmse': self.best_val_rmse,
                'suggestion': 'keep training'
            }

    def plot_curves(self, save_path: Optional[str] = None):
        """
        Plot and save training curves.

        Generates 4 subplots:
          1. Train Loss (total + breakdown)
          2. Validation RMSE + Distance Accuracy
          3. Validation Direction Accuracy + Intent Accuracy
          4. Learning Rate (if recorded)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
            import matplotlib.pyplot as plt
            # Fix CJK font: use a font that supports ASCII at minimum
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        except ImportError:
            print("[Logger] matplotlib not installed. Skipping plot. pip install matplotlib")
            return

        if not self.history:
            print("[Logger] No data to plot.")
            return

        epochs = [e['epoch'] for e in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('EMAM Training Curves', fontsize=14, fontweight='bold')

        # --- Subplot 1: Train Loss ---
        ax = axes[0, 0]
        ax.plot(epochs, [e['train_loss'] for e in self.history],
                'b-', linewidth=1, alpha=0.7, label='Total Loss')
        # Smooth curve
        if len(self.history) >= self.smoothing_window:
            smoothed = self._smooth([e['train_loss'] for e in self.history])
            ax.plot(epochs, smoothed, 'b-', linewidth=2, label='Total Loss (smoothed)')

        # Breakdown
        colors = {'displacement': 'orange', 'intent': 'green',
                  'uncertainty': 'red', 'physics': 'purple'}
        for key, color in colors.items():
            values = [e.get(f'train_loss_{key}', None) for e in self.history]
            if any(v is not None for v in values):
                ax.plot(epochs, values, '--', color=color, linewidth=0.8, alpha=0.6,
                        label=f'{key.capitalize()} Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # log scale for better visualization

        # --- Subplot 2: Validation RMSE + Distance Accuracy ---
        ax = axes[0, 1]
        val_epochs = [e['epoch'] for e in self.history if 'val_rmse' in e]
        if val_epochs:
            rmse_vals = [e['val_rmse'] for e in self.history if 'val_rmse' in e]
            ax.plot(val_epochs, rmse_vals, 'r-o', markersize=4, linewidth=1.5, label='RMSE (m)')

            # Mark best epoch
            if self.best_epoch in val_epochs:
                best_idx = val_epochs.index(self.best_epoch)
                ax.plot(self.best_epoch, rmse_vals[best_idx], 'r*', markersize=15,
                        label=f'Best (epoch {self.best_epoch}, RMSE={self.best_val_rmse:.4f})')

            ax_twin = ax.twinx()
            dist_acc = [e['val_dist_acc'] for e in self.history if 'val_dist_acc' in e]
            ax_twin.plot(val_epochs, dist_acc, 'b--o', markersize=4, linewidth=1.5,
                         label='Distance Acc')
            ax_twin.set_ylabel('Distance Accuracy', color='b')
            ax_twin.tick_params(axis='y', labelcolor='b')
            ax_twin.legend(fontsize=7, loc='lower right')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (m)', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        ax.set_title('Validation RMSE & Distance Accuracy')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Subplot 3: Validation Direction + Intent Accuracy ---
        ax = axes[1, 0]
        if val_epochs:
            dir_acc = [e.get('val_dir_acc', None) for e in self.history if 'val_dir_acc' in e]
            intent_acc = [e.get('val_intent_acc', None) for e in self.history if 'val_intent_acc' in e]

            if any(v is not None for v in dir_acc):
                ax.plot(val_epochs, dir_acc, 'g-o', markersize=4, linewidth=1.5,
                        label='Direction Accuracy')
            if any(v is not None for v in intent_acc):
                ax.plot(val_epochs, intent_acc, 'm-s', markersize=4, linewidth=1.5,
                        label='Intent Accuracy')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Direction & Intent Accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # --- Subplot 4: Learning Rate ---
        ax = axes[1, 1]
        lr_values = [e.get('lr', None) for e in self.history]
        if any(v is not None for v in lr_values):
            ax.plot(epochs, lr_values, 'k-', linewidth=1.5)
            ax.set_yscale('log')
            ax.set_ylabel('Learning Rate')
        else:
            ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)

        # Convergence annotation
        conv = self.check_convergence()
        conv_label = conv.get('message', '')
        ax.set_xlabel('Epoch')
        ax.set_title(f'LR Schedule | {conv_label}')
        ax.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        save_path = save_path or str(self.log_dir / 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Logger] Curves saved to {save_path}")

    def _smooth(self, values: List[float]) -> np.ndarray:
        """Moving-average smoothing."""
        if len(values) < self.smoothing_window:
            return np.array(values)
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(values, kernel, mode='same')

    def get_summary(self) -> Dict:
        """Get training summary."""
        if not self.history:
            return {'status': 'no data'}

        last = self.history[-1]
        conv = self.check_convergence()

        return {
            'total_epochs': len(self.history),
            'best_epoch': self.best_epoch,
            'best_val_rmse': self.best_val_rmse,
            'current_train_loss': last.get('train_loss', float('nan')),
            'current_val_rmse': last.get('val_rmse', float('nan')),
            'current_val_dist_acc': last.get('val_dist_acc', float('nan')),
            'convergence': conv,
        }
