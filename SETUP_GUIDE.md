# Setup and Installation Guide

## Quick Start (5 minutes)

### 1. Clone or Download Repository

```bash
# If using Git
git clone <your-repo-url>
cd parametric-curve-fitting

# Or extract from ZIP and navigate to folder
```

### 2. Setup Python Environment

**Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n curve-fitting python=3.9
conda activate curve-fitting
pip install -r requirements.txt
```

### 3. Prepare Data

Place your `xy_data.csv` file in the `data/` directory:

```bash
mkdir -p data
# Copy xy_data.csv to data/ folder
cp /path/to/xy_data.csv data/
```

### 4. Run Optimization

```bash
# Simple run
python src/optimizer.py

# With visualization
python src/optimizer.py --visualize

# With custom data file
python src/optimizer.py --data data/xy_data.csv --output results
```

### 5. View Results

```bash
# View parameters
cat results/parameters.json

# View summary
cat results/summary.txt

# Open plots (if visualization was enabled)
open results/fitted_curve.png
open results/residuals.png
open results/convergence.png
```

## Detailed Installation

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for environment and results
- **OS**: Linux, macOS, or Windows

### Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy>=1.21.0         # Numerical computing
pandas>=1.3.0         # Data manipulation
scipy>=1.7.0          # Scientific computing & optimization
matplotlib>=3.4.0     # Plotting
seaborn>=0.11.0      # Statistical visualization
jupyter>=1.0.0        # Interactive notebooks
pytest>=7.0.0         # Testing framework
```

### Installation Steps

1. **Install Python 3.8+**
   
   Check your Python version:
   ```bash
   python3 --version
   ```
   
   If not installed, download from [python.org](https://www.python.org/downloads/)

2. **Create Project Directory Structure**
   
   ```bash
   mkdir -p parametric-curve-fitting
   cd parametric-curve-fitting
   mkdir -p data results src notebooks tests
   ```

3. **Install Dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
   
   If you encounter issues:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   
   ```bash
   python -c "import numpy, pandas, scipy, matplotlib; print('All packages installed successfully!')"
   ```

## Project Structure

```
parametric-curve-fitting/
├── README.md                   # Main documentation
├── SETUP_GUIDE.md             # This file
├── requirements.txt           # Python dependencies
├── run_optimization.sh        # Quick start script
│
├── data/
│   └── xy_data.csv           # Input data (you provide)
│
├── src/
│   ├── optimizer.py          # Main optimization code
│   ├── visualizer.py         # Visualization functions
│   └── utils.py              # Utility functions
│
├── results/                   # Generated outputs
│   ├── parameters.json       # Optimal parameters
│   ├── fitted_curve.png      # Visualization
│   ├── residuals.png         # Error analysis
│   ├── convergence.png       # Optimization progress
│   └── summary.txt           # Detailed report
│
├── notebooks/
│   └── analysis.ipynb        # Jupyter notebook for analysis
│
└── tests/
    └── test_optimizer.py     # Unit tests
```

## Usage Examples

### Basic Usage

```python
from src.optimizer import ParametricCurveOptimizer
from src.utils import load_data

# Load data
data = load_data('data/xy_data.csv')

# Create optimizer
optimizer = ParametricCurveOptimizer()

# Fit parameters
result = optimizer.fit(data)

# Print results
print(f"θ = {result['theta']:.6f}°")
print(f"M = {result['M']:.8f}")
print(f"X = {result['X']:.2f}")
```

### With Custom Parameter Ranges

```python
optimizer = ParametricCurveOptimizer(
    theta_range=(10, 40),  # Custom theta range
    M_range=(-0.01, 0.01),  # Tighter M range
    X_range=(20, 80)        # Custom X range
)

result = optimizer.fit(data)
```

### Generate Only Visualization

```python
from src.visualizer import generate_plots

# After fitting
generate_plots(data, optimizer, output_dir='results')
```

### Export Results

```python
from src.utils import export_to_latex, create_summary_report

# Export to LaTeX
export_to_latex(result, 'results/solution.tex')

# Create summary report
create_summary_report(result, data, optimizer, 'results/summary.txt')
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_optimizer.py::TestParametricCurveOptimizer::test_compute_curve
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Permission denied when running scripts

**Solution:**
```bash
chmod +x run_optimization.sh
./run_optimization.sh
```

### Issue: matplotlib backend errors

**Solution:**
Add this to the beginning of your script:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

### Issue: Out of memory errors

**Solution:**
Reduce grid search resolution:
```python
result = optimizer.grid_search(data, theta_steps=10, M_steps=10, X_steps=10)
```

### Issue: Optimization takes too long

**Solution:**
Use faster mode:
```python
result = optimizer.fit(data, use_all_stages=False)
```

## Performance Tips

1. **Use smaller grid for initial testing:**
   ```python
   optimizer.grid_search(data, theta_steps=15, M_steps=15, X_steps=15)
   ```

2. **Skip visualization for faster runs:**
   ```bash
   python src/optimizer.py --data data/xy_data.csv
   ```

3. **Use multiprocessing (advanced):**
   Modify the grid search to use `multiprocessing.Pool` for parallel evaluation.

## Advanced Configuration

### Custom Objective Function

You can modify the L1 distance calculation in `optimizer.py`:

```python
def compute_L2_distance(self, params, data, t_values=None):
    """Use L2 (Euclidean) distance instead."""
    theta, M, X = params
    
    if t_values is None:
        n_points = len(data)
        t_values = np.linspace(6, 60, n_points)
    
    x_pred, y_pred = self.compute_curve(theta, M, X, t_values)
    
    x_obs = data['x'].values
    y_obs = data['y'].values
    
    distance = np.sqrt(np.mean((x_obs - x_pred)**2 + (y_obs - y_pred)**2))
    
    return distance
```

### Add Additional Optimization Stages

```python
def custom_stage(self, data, initial_params):
    # Your custom optimization logic
    return optimized_params, distance

# Add to fit method
result = optimizer.fit(data)
custom_result = optimizer.custom_stage(data, result['params'])
```

## Getting Help

1. **Check the README.md** for methodology details
2. **Review test cases** in `tests/test_optimizer.py`
3. **Examine the code** - it's well documented
4. **Run with verbose output:**
   ```bash
   python src/optimizer.py --verbose
   ```

## Next Steps

1. ✅ Complete installation
2. ✅ Run optimization on sample data
3. ✅ Review results and visualizations
4. ✅ Copy Desmos format for submission
5. ✅ Document your approach (for assessment)
6. ✅ Submit to GitHub repository

## Additional Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **SciPy Optimization**: https://docs.scipy.org/doc/scipy/reference/optimize.html
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Desmos Calculator**: https://www.desmos.com/calculator

## License

MIT License - Feel free to use and modify as needed.