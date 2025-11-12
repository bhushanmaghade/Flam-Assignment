"""
Visualization module for parametric curve fitting results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def generate_plots(data, optimizer, output_dir='results'):
    """
    Generate all visualization plots.
    
    Args:
        data: DataFrame with observed points
        optimizer: ParametricCurveOptimizer instance with fitted parameters
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Generate plots
    plot_fitted_curve(data, optimizer, output_path)
    plot_residuals(data, optimizer, output_path)
    plot_convergence(optimizer, output_path)
    
    print(f"Plots saved to {output_dir}/")


def plot_fitted_curve(data, optimizer, output_path):
    """Plot observed data vs fitted curve."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get optimal parameters
    theta, M, X = optimizer.best_params
    
    # Generate fine-grained curve for plotting
    t_fine = np.linspace(6, 60, 1000)
    x_pred_fine, y_pred_fine = optimizer.compute_curve(theta, M, X, t_fine)
    
    # Generate predictions at data points
    t_data = np.linspace(6, 60, len(data))
    x_pred_data, y_pred_data = optimizer.compute_curve(theta, M, X, t_data)
    
    # Plot observed data
    ax.scatter(data['x'], data['y'], c='red', s=20, alpha=0.5, 
              label='Observed Data', zorder=3)
    
    # Plot fitted curve
    ax.plot(x_pred_fine, y_pred_fine, 'b-', linewidth=2, 
           label='Fitted Curve', zorder=2)
    
    # Plot predicted points at data locations
    ax.scatter(x_pred_data, y_pred_data, c='green', s=10, alpha=0.3,
              label='Predicted Points', zorder=1, marker='x')
    
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_title('Parametric Curve Fitting: Observed vs Predicted', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add parameter text box
    param_text = (f'θ = {theta:.6f}°\n'
                 f'M = {M:.8f}\n'
                 f'X = {X:.2f}\n'
                 f'L1 Distance = {optimizer.best_distance:.6f}')
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'fitted_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(data, optimizer, output_path):
    """Plot residual errors."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get optimal parameters
    theta, M, X = optimizer.best_params
    
    # Generate predictions
    t_data = np.linspace(6, 60, len(data))
    x_pred, y_pred = optimizer.compute_curve(theta, M, X, t_data)
    
    # Calculate residuals
    x_residuals = data['x'].values - x_pred
    y_residuals = data['y'].values - y_pred
    total_residuals = np.abs(x_residuals) + np.abs(y_residuals)
    
    # Plot 1: X residuals vs t
    axes[0, 0].scatter(t_data, x_residuals, c='blue', alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('t parameter', fontsize=12)
    axes[0, 0].set_ylabel('X Residual (observed - predicted)', fontsize=12)
    axes[0, 0].set_title('X-Coordinate Residuals', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y residuals vs t
    axes[0, 1].scatter(t_data, y_residuals, c='green', alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('t parameter', fontsize=12)
    axes[0, 1].set_ylabel('Y Residual (observed - predicted)', fontsize=12)
    axes[0, 1].set_title('Y-Coordinate Residuals', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
    axes[1, 0].hist(total_residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(total_residuals), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(total_residuals):.4f}')
    axes[1, 0].set_xlabel('L1 Distance per Point', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Q-Q plot for normality check
    from scipy import stats
    stats.probplot(total_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Residual Normality', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = (f'Statistics:\n'
                 f'Mean L1: {np.mean(total_residuals):.6f}\n'
                 f'Std L1: {np.std(total_residuals):.6f}\n'
                 f'Max L1: {np.max(total_residuals):.6f}\n'
                 f'Min L1: {np.min(total_residuals):.6f}\n'
                 f'Median L1: {np.median(total_residuals):.6f}')
    
    fig.text(0.99, 0.01, stats_text, transform=fig.transFigure,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence(optimizer, output_path):
    """Plot optimization convergence history."""
    if not optimizer.optimization_history:
        print("No optimization history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract history
    stages = [h['stage'] for h in optimizer.optimization_history]
    distances = [h['distance'] for h in optimizer.optimization_history]
    
    # Extract parameter evolution
    thetas = [h['params'][0] for h in optimizer.optimization_history]
    Ms = [h['params'][1] for h in optimizer.optimization_history]
    Xs = [h['params'][2] for h in optimizer.optimization_history]
    
    stage_indices = list(range(len(stages)))
    
    # Plot 1: L1 Distance convergence
    axes[0, 0].plot(stage_indices, distances, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 0].set_xlabel('Optimization Stage', fontsize=12)
    axes[0, 0].set_ylabel('L1 Distance', fontsize=12)
    axes[0, 0].set_title('Convergence: L1 Distance', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(stage_indices)
    axes[0, 0].set_xticklabels([s.replace('_', ' ').title() for s in stages], 
                              rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Theta convergence
    axes[0, 1].plot(stage_indices, thetas, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 1].set_xlabel('Optimization Stage', fontsize=12)
    axes[0, 1].set_ylabel('θ (degrees)', fontsize=12)
    axes[0, 1].set_title('Convergence: θ Parameter', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(stage_indices)
    axes[0, 1].set_xticklabels([s.replace('_', ' ').title() for s in stages], 
                              rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: M convergence
    axes[1, 0].plot(stage_indices, Ms, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Optimization Stage', fontsize=12)
    axes[1, 0].set_ylabel('M', fontsize=12)
    axes[1, 0].set_title('Convergence: M Parameter', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(stage_indices)
    axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in stages], 
                              rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: X convergence
    axes[1, 1].plot(stage_indices, Xs, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Optimization Stage', fontsize=12)
    axes[1, 1].set_ylabel('X', fontsize=12)
    axes[1, 1].set_title('Convergence: X Parameter', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(stage_indices)
    axes[1, 1].set_xticklabels([s.replace('_', ' ').title() for s in stages], 
                              rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_fitted_curve_data(data, optimizer, output_path):
    """Save fitted curve data to CSV."""
    theta, M, X = optimizer.best_params
    
    # Generate fine curve
    t_fine = np.linspace(6, 60, 1000)
    x_pred, y_pred = optimizer.compute_curve(theta, M, X, t_fine)
    
    # Create DataFrame
    fitted_df = pd.DataFrame({
        't': t_fine,
        'x': x_pred,
        'y': y_pred
    })
    
    # Save to CSV
    fitted_df.to_csv(output_path / 'fitted_curve.csv', index=False)
    print(f"Fitted curve data saved to {output_path}/fitted_curve.csv")