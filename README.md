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
from incremental_softmax_recovery import incremental_softmax_recovery

# Your data must follow the sequential format
df = pd.DataFrame({
    'case:concept:name': ['0', '0', '1', '1', '2', '2'],
    'concept:name': ['0', '1', '0', '2', '1', '2']
})

# Softmax matrices aligned with cases
softmax_matrices = [matrix_0, matrix_1, matrix_2]  # One per case

# Run incremental recovery (conformance-only)
results_df, accuracy_dict, prob_dict = incremental_softmax_recovery(
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
| `conformance_switch_penalty_weight` | Weight on label-switch penalty | 0.0 | Uses `prob_dict` if > 0 |
| `max_hist_len` | History length for `prob_dict` | 3 | Used when switch penalty enabled |
| `n_indices` / `n_per_run` | Sampling controls | None | Required based on sampling mode |

## 📊 Output Format

Returns a tuple `(results_df, accuracy_dict, prob_dict)` where `results_df` contains:
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

## 🔧 Advanced Features

**Temperature Calibration**:
```python
results_df, accuracy_dict, prob_dict = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    use_calibration=True,
    temp_bounds=(0.5, 5.0),
    n_indices=4,
)
```

**Sequential Sampling**:
```python
results_df, accuracy_dict, prob_dict = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    sequential_sampling=True,
    n_per_run=2  # Sample 2 events from each activity run
)
```

## 📁 Module Structure

```
├── incremental_softmax_recovery.py  # Main entry point (conformance-only)
├── classes.py                       # Petri net classes
├── data_processing.py               # Data utilities
├── petri_model.py                   # Model discovery and prob_dict builder
├── calibration.py                   # Temperature scaling
├── utils.py                         # Helper functions
├── test_incremental_recovery.ipynb  # Test notebook
└── README.md                        # This file
```

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