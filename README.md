# Incremental Softmax Recovery for Long Traces

A Python system for incrementally recovering activity sequences from softmax probability matrices using conformance checking with Petri nets.

## 🎯 Overview

This system performs incremental trace recovery - predicting activity sequences step-by-step from neural network softmax outputs. It uses:

- Petri Net Discovery: Learns process models from training traces
- Conformance Checking: Deterministic, chunked recovery under Petri net constraints
- Conditional Probabilities (optional): Bigram history for label-switch penalty
- Flexible Sampling: Supports both uniform and sequential event sampling

## 📋 Requirements

**Data Format Requirements** (Critical):
- Case IDs must be sequential strings: `['0', '1', '2', ...]`
- Activity names must be sequential strings: `['0', '1', '2', ...]`  
- Softmax matrices must align with case order

**Dependencies**:
```bash
pip install pandas numpy scipy matplotlib seaborn pm4py scikit-learn
```

## 🚀 Quick Start

```python
from src.incremental_softmax_recovery import incremental_softmax_recovery

# Your data must follow the sequential format
df = pd.DataFrame({
    'case:concept:name': ['0', '0', '1', '1', '2', '2'],
    'concept:name': ['0', '1', '0', '2', '1', '2']
})

# Softmax matrices aligned with cases
softmax_matrices = [matrix_0, matrix_1, matrix_2]  # One per case

# Run incremental recovery (conformance-only)
results_df, accuracy_dict, prob_dict_uncollapsed, prob_dict_collapsed = incremental_softmax_recovery(
    df=df,
    softmax_lst=softmax_matrices,
    n_train_traces=10,
    n_test_traces=5,
    n_indices=4,     # Sample 4 events per trace (when sequential_sampling=False)
    sequential_sampling=False,
    prob_threshold=1e-6,
    chunk_size=15,
    conformance_switch_penalty_weight=1.0,
)

print(f"Final accuracy: {pd.Series(accuracy_dict['sktr_accuracy']).mean():.3f}")
```

## 🧪 Testing

Run the comprehensive test notebook:

```bash
jupyter notebook test_incremental_recovery.ipynb
```

The notebook includes:
- Synthetic data generation
- Core functionality testing  
- Error handling validation
- Performance benchmarking
- Results visualization

## ⚙️ Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|--------|
| `prob_threshold` | Minimum probability to consider activity | 1e-12 | Applies during filtering |
| `chunk_size` | Window size for conformance processing | 10 | Larger may be more accurate |
| `conformance_switch_penalty_weight` | Weight on label-switch penalty | 0.0 | Uses `prob_dict_uncollapsed` and `prob_dict_collapsed` if > 0 |
| `max_hist_len` | History length for `prob_dict` | 3 | Used when switch penalty enabled |
| `n_indices` / `n_per_run` | Sampling controls | None | Required based on sampling mode |
| `save_model_path` | Path for Petri net visualization | `./results/discovered_petri_net` | Saves PDF and PNG |
| `save_model` | Whether to save Petri net visualization | `True` | Set to `False` to skip saving |

## 📊 Output Format

Returns a tuple `(results_df, accuracy_dict, prob_dict_uncollapsed, prob_dict_collapsed)` where:

- `results_df`: DataFrame containing per-step recovery results with columns:
  - `case:concept:name`: Test case ID
  - `step`: Window-relative step (0..chunk_size-1 repeating)
  - `sktr_activity`: Conformance prediction
  - `argmax_activity`: Argmax baseline prediction
  - `ground_truth`: Actual activity
  - `all_probs`: Per-step filtered probabilities
  - `all_activities`: Activity labels for `all_probs`
  - `is_correct`: Boolean correctness
  - `cumulative_accuracy`: Running accuracy
  - `sktr_move_cost`: Per-move costs

- `accuracy_dict`: Dictionary with keys `'sktr_accuracy'` and `'argmax_accuracy'`, each containing a list of trace-level accuracies

- `prob_dict_uncollapsed`: Probability dictionary for continuation probabilities (used when staying within the same activity run). Maps history tuples to probability distributions over next activities, preserving activity runs as sequences of identical labels.

- `prob_dict_collapsed`: Probability dictionary for transition probabilities (used when switching between different activities). Maps history tuples to probability distributions over next activities, collapsing consecutive identical activities into single runs.

## 🔧 Advanced Features

**Temperature Calibration**:
```python
results_df, accuracy_dict, prob_dict_uncollapsed, prob_dict_collapsed = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    use_calibration=True,
    temp_bounds=(0.5, 5.0),
    n_indices=4,
)
```

**Sequential Sampling**:
```python
results_df, accuracy_dict, prob_dict_uncollapsed, prob_dict_collapsed = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    sequential_sampling=True,
    n_per_run=2  # Sample 2 events from each activity run
)
```

## 📁 Project Structure

```
SKTR_for_Long_Traces/
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── incremental_softmax_recovery.py  # Main entry point
│   ├── classes.py                    # Petri net classes
│   ├── data_processing.py            # Data utilities
│   ├── petri_model.py               # Model discovery and prob_dict builder
│   ├── calibration.py               # Temperature scaling
│   ├── conformance_checking.py      # Conformance checking algorithms
│   ├── evaluation.py                # Evaluation metrics
│   ├── utils.py                     # Helper functions
│   └── trace_export.py               # Trace export utilities
├── results/                          # Experiment results (CSV, PKL, PDF files)
│   ├── incremental_recovery_*.csv   # CSV result files
│   ├── sktr_*.csv                   # SKTR result files
│   ├── kari_*.csv                   # KARI comparison results
│   ├── kari_50salads_results.pkl   # Pickled results
│   └── discovered_petri_net.pdf     # Petri net visualizations
├── data/                             # Data files
│   ├── ground_truth_50salads_sequences.csv
│   └── sampled_traces.txt
├── test_incremental_recovery.ipynb   # Main test notebook
├── activity_distibution_investigation.ipynb
├── complete_traces_investigation.ipynb
├── kari.ipynb
└── README.md                         # This file
```

**Note**: All Python modules are now organized in the `src/` directory. Update your imports accordingly:
- Old: `from incremental_softmax_recovery import ...`
- New: `from src.incremental_softmax_recovery import ...`

## 📝 Citation

If you use this system in research, please cite:
```
[Your citation information here]
```

## 🤝 Contributing

1. Fork the repository
2. Run the test notebook to ensure everything works
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Add your license information here] 