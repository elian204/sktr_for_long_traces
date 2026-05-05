#!/bin/bash
#SBATCH --job-name=kfold_sktr_breakfast_mstcn2
#SBATCH --output=kfold_breakfast_mstcn2_%j.log
#SBATCH --error=kfold_breakfast_mstcn2_%j.err
#SBATCH --partition=cpu512G-48h     # 512GB RAM, 48h max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112         # Max for this partition (180 processes will time-share)
#SBATCH --mem=400G                  # 180 processes, shared data
#SBATCH --time=47:55:00             # Just under 48h limit

# --- CONFIGURATION ---
PROJECT_DIR="$HOME/sktr_for_long_traces"
SCRIPT_NAME="kfold_learning_curve_experiment.py"

# Experiment parameters - breakfast mstcn2
DATASET="breakfast"                 # Options: breakfast, 50salads, gtea
MODEL="mstcn2"                      # Options: asformer, mstcn2
WORKERS=60                          # Inner workers per experiment
PARALLEL=3                          # Number of (fold,k) experiments in parallel

echo "========================================"
echo "KFOLD LEARNING CURVE EXPERIMENT"
echo "========================================"
echo "Running on: $(hostname)"
echo "Date: $(date)"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Workers: $WORKERS"
echo "Parallel experiments: $PARALLEL"
echo "========================================"

# 1. SET DATA PATHS
# Point to where fold split files are located
export DATA_ROOT="/home/dsi/eli-bogdanov/data/data"

# 2. PREVENT THREAD OVERSUBSCRIPTION
# Stop numpy/BLAS from spawning extra threads on top of our workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 3. ACTIVATE PYTHON ENVIRONMENT
source ~/sktr_env/bin/activate

# 4. VERIFY PYTHON ENVIRONMENT
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# 5. CHECK DEPENDENCIES
echo "Checking key dependencies..."
python -c "import pandas, numpy, pm4py, joblib, seaborn, matplotlib; print('Dependencies OK')" || { echo "Missing dependencies!"; exit 1; }

# 6. RUN EXPERIMENT
cd "$PROJECT_DIR" || { echo "Cannot find project folder: $PROJECT_DIR"; exit 1; }

python -u "$SCRIPT_NAME" \
    -d "$DATASET" \
    -m "$MODEL" \
    -w "$WORKERS" \
    -p "$PARALLEL" \
    --inner-parallel \
    --alpha 0.7 \
    --strategy trigram_heavy \
    --restrict-log-moves \
    --restrict-model-moves-to-tau \
    --top-m 1 \
    --candidate-top-k 3 \
    --no-save-models \
    --skip-existing \
    --unique-only

echo "========================================"
echo "Experiment completed at: $(date)"
echo "========================================"
