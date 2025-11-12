#!/bin/bash

# Parametric Curve Optimization - Quick Start Script

echo "================================================"
echo "Parametric Curve Parameter Estimation"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Creating directory structure..."
mkdir -p data results src notebooks tests

# Copy data file if it doesn't exist
if [ ! -f "data/xy_data.csv" ]; then
    echo "Please place xy_data.csv in the data/ directory"
    exit 1
fi

echo ""
echo "Step 3: Running optimization..."
python src/optimizer.py --data data/xy_data.csv --output results --visualize

echo ""
echo "================================================"
echo "Optimization Complete!"
echo "================================================"
echo ""
echo "Results saved in the 'results/' directory:"
echo "  - parameters.json      : Optimal parameters"
echo "  - fitted_curve.png     : Visual comparison"
echo "  - residuals.png        : Error analysis"
echo "  - convergence.png      : Optimization progress"
echo "  - summary.txt          : Detailed report"
echo ""
echo "To view results:"
echo "  cat results/parameters.json"
echo ""