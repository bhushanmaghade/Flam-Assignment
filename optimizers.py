"""
Parametric Curve Optimizer
Finds optimal parameters (theta, M, X) for the given parametric equations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import json
import argparse
from pathlib import Path
import time


class ParametricCurveOptimizer:
    """
    Optimizer for parametric curve fitting problem.
    
    Equations:
    x = t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
    y = 42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)
    """
    
    def __init__(self, theta_range=(0, 50), M_range=(-0.05, 0.05), X_range=(0, 100)):
        """
        Initialize optimizer with parameter ranges.
        
        Args:
            theta_range: (min, max) for theta in degrees
            M_range: (min, max) for M
            X_range: (min, max) for X
        """
        self.theta_range = theta_range
        self.M_range = M_range
        self.X_range = X_range
        self.best_params = None
        self.best_distance = float('inf')
        self.optimization_history = []
        
    def compute_curve(self, theta, M, X, t_values):
        """
        Compute parametric curve points.
        
        Args:
            theta: Angle in degrees
            M: Exponential coefficient
            X: X offset
            t_values: Array of t parameter values
            
        Returns:
            x, y: Arrays of curve coordinates
        """
        theta_rad = np.deg2rad(theta)
        exp_term = np.exp(M * np.abs(t_values))
        sin_03t = np.sin(0.3 * t_values)
        
        x = (t_values * np.cos(theta_rad) - 
             exp_term * sin_03t * np.sin(theta_rad) + X)
        y = (42 + t_values * np.sin(theta_rad) + 
             exp_term * sin_03t * np.cos(theta_rad))
        
        return x, y
    
    def compute_L1_distance(self, params, data, t_values=None):
        """
        Compute L1 (Manhattan) distance between observed and predicted points.
        
        Args:
            params: [theta, M, X]
            data: DataFrame with 'x' and 'y' columns
            t_values: Optional array of t values (will be generated if None)
            
        Returns:
            L1 distance (mean absolute deviation)
        """
        theta, M, X = params
        
        # Enforce constraints
        if not (self.theta_range[0] <= theta <= self.theta_range[1]):
            return 1e10
        if not (self.M_range[0] <= M <= self.M_range[1]):
            return 1e10
        if not (self.X_range[0] <= X <= self.X_range[1]):
            return 1e10
        
        # Generate uniform t values matching data length
        if t_values is None:
            n_points = len(data)
            t_values = np.linspace(6, 60, n_points)
        
        # Compute predicted curve
        x_pred, y_pred = self.compute_curve(theta, M, X, t_values)
        
        # Extract observed points
        x_obs = data['x'].values
        y_obs = data['y'].values
        
        # Compute L1 distance
        distance = np.mean(np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred))
        
        return distance
    
    def grid_search(self, data, theta_steps=20, M_steps=20, X_steps=20):
        """
        Perform coarse grid search over parameter space.
        
        Args:
            data: DataFrame with observed points
            theta_steps: Number of grid points for theta
            M_steps: Number of grid points for M
            X_steps: Number of grid points for X
            
        Returns:
            best_params: [theta, M, X] with minimum distance
        """
        print("Stage 1: Coarse Grid Search...")
        
        theta_grid = np.linspace(self.theta_range[0], self.theta_range[1], theta_steps)
        M_grid = np.linspace(self.M_range[0], self.M_range[1], M_steps)
        X_grid = np.linspace(self.X_range[0], self.X_range[1], X_steps)
        
        best_distance = float('inf')
        best_params = None
        
        total_iterations = theta_steps * M_steps * X_steps
        iteration = 0
        
        for theta in theta_grid:
            for M in M_grid:
                for X in X_grid:
                    iteration += 1
                    if iteration % 1000 == 0:
                        print(f"  Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")
                    
                    params = [theta, M, X]
                    distance = self.compute_L1_distance(params, data)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_params = params
                        print(f"  New best: θ={theta:.2f}°, M={M:.6f}, X={X:.2f}, L1={distance:.6f}")
        
        self.optimization_history.append({
            'stage': 'grid_search',
            'params': best_params,
            'distance': best_distance
        })
        
        return best_params, best_distance
    
    def local_search(self, data, initial_params, search_radius=0.1, steps=15):
        """
        Perform local grid search around initial parameters.
        
        Args:
            data: DataFrame with observed points
            initial_params: [theta, M, X] starting point
            search_radius: Fraction of parameter range to search
            steps: Number of grid points in each dimension
            
        Returns:
            best_params: [theta, M, X] with minimum distance
        """
        print("\nStage 2: Local Refinement...")
        
        theta_0, M_0, X_0 = initial_params
        
        theta_radius = (self.theta_range[1] - self.theta_range[0]) * search_radius
        M_radius = (self.M_range[1] - self.M_range[0]) * search_radius
        X_radius = (self.X_range[1] - self.X_range[0]) * search_radius
        
        theta_grid = np.linspace(
            max(self.theta_range[0], theta_0 - theta_radius),
            min(self.theta_range[1], theta_0 + theta_radius),
            steps
        )
        M_grid = np.linspace(
            max(self.M_range[0], M_0 - M_radius),
            min(self.M_range[1], M_0 + M_radius),
            steps
        )
        X_grid = np.linspace(
            max(self.X_range[0], X_0 - X_radius),
            min(self.X_range[1], X_0 + X_radius),
            steps
        )
        
        best_distance = float('inf')
        best_params = initial_params
        
        for theta in theta_grid:
            for M in M_grid:
                for X in X_grid:
                    params = [theta, M, X]
                    distance = self.compute_L1_distance(params, data)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_params = params
        
        print(f"  Best: θ={best_params[0]:.4f}°, M={best_params[1]:.8f}, X={best_params[2]:.2f}, L1={best_distance:.6f}")
        
        self.optimization_history.append({
            'stage': 'local_search',
            'params': best_params,
            'distance': best_distance
        })
        
        return best_params, best_distance
    
    def gradient_optimization(self, data, initial_params):
        """
        Apply gradient-based optimization (Nelder-Mead).
        
        Args:
            data: DataFrame with observed points
            initial_params: [theta, M, X] starting point
            
        Returns:
            best_params: [theta, M, X] with minimum distance
        """
        print("\nStage 3: Gradient-Based Optimization...")
        
        # Define bounds
        bounds = [
            self.theta_range,
            self.M_range,
            self.X_range
        ]
        
        # Optimize using Nelder-Mead
        result = minimize(
            lambda p: self.compute_L1_distance(p, data),
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        best_params = result.x
        best_distance = result.fun
        
        print(f"  Best: θ={best_params[0]:.4f}°, M={best_params[1]:.8f}, X={best_params[2]:.2f}, L1={best_distance:.6f}")
        print(f"  Converged: {result.success}, Iterations: {result.nit}")
        
        self.optimization_history.append({
            'stage': 'gradient_optimization',
            'params': best_params.tolist(),
            'distance': best_distance
        })
        
        return best_params, best_distance
    
    def random_perturbation(self, data, initial_params, iterations=5000):
        """
        Apply random perturbations to escape local minima.
        
        Args:
            data: DataFrame with observed points
            initial_params: [theta, M, X] starting point
            iterations: Number of random perturbations to try
            
        Returns:
            best_params: [theta, M, X] with minimum distance
        """
        print("\nStage 4: Random Perturbation Search...")
        
        best_params = initial_params
        best_distance = self.compute_L1_distance(initial_params, data)
        
        theta_0, M_0, X_0 = initial_params
        
        improvements = 0
        
        for i in range(iterations):
            # Decreasing perturbation magnitude
            scale = 1.0 / (1 + i / 1000)
            
            # Random perturbations
            theta_perturb = theta_0 + np.random.uniform(-5, 5) * scale
            M_perturb = M_0 + np.random.uniform(-0.01, 0.01) * scale
            X_perturb = X_0 + np.random.uniform(-10, 10) * scale
            
            params = [theta_perturb, M_perturb, X_perturb]
            distance = self.compute_L1_distance(params, data)
            
            if distance < best_distance:
                best_distance = distance
                best_params = params
                theta_0, M_0, X_0 = params
                improvements += 1
                
                if improvements % 10 == 0:
                    print(f"  Improvement #{improvements}: L1={best_distance:.6f}")
        
        print(f"  Total improvements: {improvements}")
        print(f"  Final: θ={best_params[0]:.4f}°, M={best_params[1]:.8f}, X={best_params[2]:.2f}, L1={best_distance:.6f}")
        
        self.optimization_history.append({
            'stage': 'random_perturbation',
            'params': best_params,
            'distance': best_distance
        })
        
        return best_params, best_distance
    
    def fit(self, data, use_all_stages=True):
        """
        Main optimization routine using all stages.
        
        Args:
            data: DataFrame with 'x' and 'y' columns
            use_all_stages: Whether to use all optimization stages
            
        Returns:
            Dictionary with optimal parameters and distance
        """
        start_time = time.time()
        
        print(f"Starting optimization with {len(data)} data points...")
        print(f"Parameter ranges:")
        print(f"  θ: {self.theta_range[0]}° to {self.theta_range[1]}°")
        print(f"  M: {self.M_range[0]} to {self.M_range[1]}")
        print(f"  X: {self.X_range[0]} to {self.X_range[1]}")
        print()
        
        # Stage 1: Grid Search
        params, distance = self.grid_search(data)
        
        if use_all_stages:
            # Stage 2: Local Refinement
            params, distance = self.local_search(data, params)
            
            # Stage 3: Gradient Optimization
            params, distance = self.gradient_optimization(data, params)
            
            # Stage 4: Random Perturbation
            params, distance = self.random_perturbation(data, params)
        
        self.best_params = params
        self.best_distance = distance
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Optimal Parameters:")
        print(f"  θ (theta) = {params[0]:.8f}°")
        print(f"  M         = {params[1]:.10f}")
        print(f"  X         = {params[2]:.8f}")
        print(f"\nPerformance:")
        print(f"  L1 Distance = {distance:.8f}")
        print(f"  Time Elapsed = {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        return {
            'theta': params[0],
            'M': params[1],
            'X': params[2],
            'L1_distance': distance,
            'time_elapsed': elapsed_time,
            'optimization_history': self.optimization_history
        }
    
    def generate_desmos_format(self):
        """Generate LaTeX format for Desmos."""
        if self.best_params is None:
            return None
        
        theta, M, X = self.best_params
        
        latex = (f"\\left(t\\cos({theta:.6f})-e^{{{M:.6f}|t|}}"
                f"\\cdot\\sin(0.3t)\\sin({theta:.6f})+{X:.2f},"
                f"42+t\\sin({theta:.6f})+e^{{{M:.6f}|t|}}"
                f"\\cdot\\sin(0.3t)\\cos({theta:.6f})\\right)")
        
        return latex


def load_data(filepath):
    """Load CSV data."""
    data = pd.read_csv(filepath)
    return data


def save_results(result, output_dir='results'):
    """Save optimization results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save parameters as JSON
    params_file = output_path / 'parameters.json'
    with open(params_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Parametric Curve Optimizer')
    parser.add_argument('--data', type=str, default='data/xy_data.csv',
                      help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} data points\n")
    
    # Create optimizer
    optimizer = ParametricCurveOptimizer()
    
    # Fit parameters
    result = optimizer.fit(data)
    
    # Generate Desmos format
    desmos_format = optimizer.generate_desmos_format()
    result['desmos_format'] = desmos_format
    
    print("Desmos Format:")
    print(desmos_format)
    print()
    
    # Save results
    save_results(result, args.output)
    
    # Generate visualization if requested
    if args.visualize:
        try:
            from visualizer import generate_plots
            print("Generating plots...")
            generate_plots(data, optimizer, args.output)
        except ImportError:
            print("Visualizer module not found. Skipping plots.")


if __name__ == '__main__':
    main()