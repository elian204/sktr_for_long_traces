#!/bin/bash
#SBATCH --job-name=kfold_sktr
#SBATCH --output=kfold_%j.log
#SBATCH --error=kfold_%j.err
#SBATCH --partition=cpu512G-48h     # 512GB RAM, 48h max - resubmit with --skip-existing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80          # Should be >= PARALLEL * WORKERS (6*10=60)
#SBATCH --mem=240G                  # 60 processes * 3GB = 180GB + reserve
#SBATCH --time=47:55:00             # Just under 48h limit

# --- CONFIGURATION ---
PROJECT_DIR="$HOME/sktr_for_long_traces"
SCRIPT_NAME="kfold_learning_curve_experiment.py"

# Experiment parameters - MODIFY THESE AS NEEDED
DATASET="50salads"                  # Options: breakfast, 50salads, gtea
MODEL="asformer"                    # Options: asformer, mstcn2
WORKERS=10                          # Inner workers per experiment
PARALLEL=6                          # Number of (fold,k) experiments in parallel

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
    --restrict-log-moves \
    --top-m 1 \
    --candidate-top-k 3 \
    --no-save-models \
    --skip-existing

echo "========================================"
echo "Experiment completed at: $(date)"
echo "========================================"
