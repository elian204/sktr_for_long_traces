# Incremental Softmax Recovery for Long Traces

A Python system for incrementally recovering activity sequences from softmax probability matrices using beam search with Petri nets.

## 🎯 Overview

This system performs **incremental trace recovery** - predicting activity sequences step-by-step from neural network softmax outputs. It uses:

- **Petri Net Discovery**: Learns process models from training traces
- **Beam Search**: Maintains multiple candidate paths for robust prediction  
- **Conditional Probabilities**: Incorporates sequence history for better accuracy
- **Flexible Sampling**: Supports both uniform and sequential event sampling

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

# Run incremental recovery
results = incremental_softmax_recovery(
    df=df,
    softmax_lst=softmax_matrices,
    n_train_traces=10,
    n_test_traces=5,
    n_indices=4,  # Sample 4 events per trace
    beam_width=10,
    random_seed=42
)

print(f"Final accuracy: {results.groupby('case:concept:name')['cumulative_accuracy'].last().mean():.3f}")
```

## 🧪 Testing

Run the comprehensive test notebook:

```bash
jupyter notebook test_incremental_recovery.ipynb
```

The notebook includes:
- ✅ Synthetic data generation
- ✅ Core functionality testing  
- ✅ Error handling validation
- ✅ Performance benchmarking
- ✅ Results visualization

## ⚙️ Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|--------|
| `beam_width` | Number of candidates to maintain | 10 | Higher = better accuracy, slower |
| `n_indices` | Events to sample per trace | None | Required for uniform sampling |
| `n_per_run` | Events per activity run | None | Required for sequential sampling |
| `use_cond_probs` | Enable conditional probabilities | False | Improves accuracy |
| `alpha` | History vs base probability weight | 0.5 | 0=history only, 1=base only |

## 📊 Output Format

Returns a DataFrame with:
- `case:concept:name`: Test case ID
- `step`: Event position in sequence  
- `predicted_activity`: Beam search prediction
- `ground_truth`: Actual activity
- `is_correct`: Boolean correctness
- `cumulative_accuracy`: Running accuracy

## 🔧 Advanced Features

**Conditional Probabilities**:
```python
results = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    use_cond_probs=True,
    max_hist_len=3,
    alpha=0.7,
    use_ngram_smoothing=True
)
```

**Temperature Calibration**:
```python
results = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    use_calibration=True,
    temp_bounds=(0.5, 5.0)
)
```

**Sequential Sampling**:
```python
results = incremental_softmax_recovery(
    df=df, softmax_lst=softmax_list,
    sequential_sampling=True,
    n_per_run=2  # Sample 2 events from each activity run
)
```

## 🐛 Bug Status

**✅ Code Verification Complete**
- No critical bugs found
- Comprehensive input validation
- Proper state management
- Good error handling

**Minor Notes**:
- Comment mismatch in line 302 (cosmetic only)
- Strict data format requirements (well documented)

## 📁 Module Structure

```
├── incremental_softmax_recovery.py  # Main entry point
├── beam_search.py                   # Beam search algorithm
├── classes.py                       # Petri net classes
├── data_processing.py               # Data utilities
├── petri_model.py                   # Model discovery
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