# IF3270_Tubes1_Machine-Love

![Banner](banner.jpeg)

## Description

This repository implements a custom Feedforward Neural Network (FFNN) from scratch in Python, including automatic differentiation, various optimization algorithms, and regularization techniques. It is developed as part of the IF3270 (Artificial Intelligence) course assignment at Institut Teknologi Bandung.

The project demonstrates core machine learning concepts such as backpropagation, gradient descent, and neural network training, without relying on high-level libraries like TensorFlow or PyTorch for the core implementation.

## Features

- **Custom FFNN Implementation**: Supports configurable hidden layers, activation functions (linear, ReLU, sigmoid, tanh, softmax, sign, softplus), and loss functions (MSE, BCE, CCE).
- **Automatic Differentiation**: Built-in autodiff engine for computing gradients.
- **Optimizers**: Gradient Descent and Adam optimizer with configurable learning rates, momentum, and RMS gain.
- **Regularization**: L1/L2 regularization and RMS normalization.
- **Weight Initialization**: Multiple initialization methods (zero, uniform, normal, Xavier, He).
- **Training History**: Tracks and visualizes loss, weights, and gradients over epochs.
- **Save/Load Functionality**: Complete model serialization including training state and history.
- **Visualization**: Methods for plotting loss curves and weight/gradient distributions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/IF3270_Tubes1_Machine-Love.git
   cd IF3270_Tubes1_Machine-Love
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

The main demonstration is in the Jupyter notebook:

```bash
jupyter notebook src/Notebook/Tubes1.ipynb
```

This notebook shows how to:
- Load and preprocess the dataset
- Initialize and train the FFNN model
- Evaluate performance
- Visualize training progress

### Using the FFNN Class Directly

```python
from src.Algorithm.ffnn import FFNN
import numpy as np

# Create model
model = FFNN(
    hidden_layer_sizes=[64, 32],
    hidden_layer_activations=['relu', 'relu'],
    output_layer_activation='softmax',
    loss_function='cce',
    optimizer='adams',
    learning_rate=0.001,
    epochs=10
)

# Train
X_train = np.loadtxt('dataset/train.csv', delimiter=',')
y_train = np.loadtxt('dataset/train_labels.csv', delimiter=',')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Save model
model.save('model.npz')

# Load model
loaded_model = FFNN.load('model.npz')
```

## Dataset

The `dataset/` folder contains:
- `train.csv`: Training features
- `test.csv`: Test features
- `dataset.csv`: Full dataset
- `custom_ffnn_weights.npz`: Pre-trained weights (optional)

## Project Structure

```
IF3270_Tubes1_Machine-Love/
├── src/
│   ├── Algorithm/
│   │   ├── ffnn.py          # Main FFNN class
│   │   ├── autodiff.py      # Automatic differentiation
│   │   └── optimizer.py     # Optimizers
│   └── Notebook/
│       └── Tubes1.ipynb     # Main notebook
├── dataset/                 # Dataset files
├── requirements.txt         # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Jupyter Notebook
- Other dependencies listed in `requirements.txt`

## Contributing

This is an academic project. For questions or improvements, please contact the repository maintainer.

## License

This project is for educational purposes as part of the IF3270 course.