#!/bin/bash
#SBATCH --job-name=kfold_sktr_gtea_asformer
#SBATCH --output=kfold_gtea_asformer_%j.log
#SBATCH --error=kfold_gtea_asformer_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # >= PARALLEL * WORKERS (5*7=35)
#SBATCH --mem=120G
#SBATCH --time=47:55:00

# --- CONFIGURATION ---
PROJECT_DIR="$HOME/sktr_for_long_traces"
SCRIPT_NAME="kfold_learning_curve_experiment.py"

# Experiment parameters - gtea asformer
DATASET="gtea"                      # Options: breakfast, 50salads, gtea
MODEL="asformer"                    # Options: asformer, mstcn2
WORKERS=7                           # Inner workers per experiment
PARALLEL=5                          # Parallel (fold,k) jobs
K_VALUES="1,5,10,15,21"

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
export DATA_ROOT="/home/dsi/eli-bogdanov/data/data"

# 2. PREVENT THREAD OVERSUBSCRIPTION
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
    --k-values "$K_VALUES" \
    --inner-parallel \
    --alpha 0.95 \
    --strategy trigram_heavy \
    --restrict-log-moves \
    --top-m 1 \
    --candidate-top-k 3 \
    --no-save-models \
    --skip-existing

echo "========================================"
echo "Experiment completed at: $(date)"
echo "========================================"
