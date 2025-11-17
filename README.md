# MLDiff

## Overview

MLDiff provides a novel approach to comparing learned classifiers by translating them into SMT formulas and systematically analyzing their decision boundaries. This enables precise identification of input regions where classifiers agree or disagree, offering insights into model behavior and reliability. The repository has the following structure:

```
MLDiff/
├── mldiff/                # Core library
│   ├── diff.py            # Main comparison framework
│   ├── dt2smt.py          # Decision Tree to SMT translation
│   ├── svm2smt.py         # SVM to SMT translation
│   ├── logReg2smt.py      # Logistic Regression to SMT translation
│   ├── mlp2smt.py         # MLP to SMT translation
│   └── ...                # Additional translation
├── examples/              # Usage examples
│   ├── iris_diff.py       # Iris dataset comparison
│   ├── digits_diff.py     # Digits dataset comparison
│   └── ...                # Additional examples
├── sefm2025/              # SEFM 2025 paper
│   ├── rq1.py             # Research Question 1
│   ├── rq2-4.py           # Research Questions 2-4
│   └── results/           # Experimental results for SEFM 2025 paper
│   └── ...                # Additional scripts
├── tests/                 # Test suite
├── models/                # Pre-trained models
└── pyproject.toml         # Project configuration
```

### Key Features

- **Multi-Classifier Support**: Compare Decision Trees, SVMs, Logistic Regression, and Multi-Layer Perceptrons
- **SMT-Based Analysis**: Leverages Z3 solver for formal verification and exhaustive comparison
- **Comprehensive Evaluation**: Generates difference matrices showing all possible agreement/disagreement patterns
- **Research Framework**: Includes experimental setup for empirical studies on classifier comparison
- **Multiple Datasets**: Built-in support for Iris, Digits, Breast Cancer, CIFAR, and Olivetti Faces datasets

## Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/yourusername/MLDiff.git
cd MLDiff
poetry install
poetry shell
```

### Using pip

```bash
git clone https://github.com/yourusername/MLDiff.git
cd MLDiff
pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- scikit-learn 1.4.2
- NumPy, Matplotlib, z3-solver
- Additional dependencies listed in `pyproject.toml`

## Quick Start

Here's a simple example comparing a Decision Tree and SVM on the Iris dataset:

```python
from z3 import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import mldiff.dt2smt as dt2smt
import mldiff.svm2smt as svm2smt
from mldiff.diff_matrix import evaluateDiffMatrix

# Load data and train classifiers
iris = load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

svm = LinearSVC()
svm.fit(X, y)

# Convert to SMT formulas
clDt = Int("classDT")
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

# Check for disagreements
s.push()
s.add(clDt != clSvm)
if s.check() == sat:
    print("Classifiers disagree on:", s.model())
else:
    print("Classifiers agree on all inputs")
s.pop()
```

## Usage Examples

The `examples/` directory contains comprehensive usage examples:

- `iris_diff.py`: Basic classifier comparison on Iris dataset
- `digits_diff.py`: Comparison on handwritten digit classification
- `cifar_diff_dt_svm.py`: CIFAR-10 dataset comparison
- `olivetti_diff_dt_svm.py`: Face recognition comparison

Run any example:

```bash
python -m examples.iris_diff
```

## Research Framework

The `sefm2025/` directory contains evaluation artifacts of the paper "On the Comparison of Learned Classifiers" submitted to SEFM 2025.

## Used Datasets

- **Iris**: Classic 3-class flower classification
- **Digits**: Handwritten digit recognition (0-9)
- **Breast Cancer**: Binary cancer diagnosis
- **Olivetti Faces**: Face recognition

## Configuration

The framework supports various configuration options in `diff.py`:

- `VIZ_FLAG`: Enable visualization of results
- `PCA_FLAG`: Apply PCA preprocessing
- `FEATURE_CON_FLAG`: Enable domain constraints
- `EPSILON_*`: Tolerance parameters for different classifiers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use MLDiff in your research, please cite:

```bibtex
@inproceedings{SoaibuzzamanSEFM25,
  author       = {Soaibuzzaman and Jenny D{\"o}ring and Srinivasulu Kasi and Jan Oliver Ringert},
  editor       = {Domenico Bianculli and Elena G{\'o}mez-Mart{\'i}nez},
  title        = {On the Comparison of Learned Classifiers},
  booktitle    = {Software Engineering and Formal Methods},
  series       = {LNCS},
  volume       = {16192},
  pages        = {223--240},
  publisher    = {Springer Nature Switzerland},
  address      = {Cham},
  year         = {2026},
  url          = {https://doi.org/10.1007/978-3-032-10444-1_14},
  doi          = {10.1007/978-3-032-10444-1_14},
  isbn         = {978-3-032-10444-1},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.