import json
from pathlib import Path
from typing import Literal
from optimizer import *
from autodiff import *
import numpy as np

class FFNN:
    "Feedforward Neural Network implementation."
    
    def __init__(
            self,
            hidden_layer_sizes: None | list[int] = None,
            hidden_layer_activations: None | list[Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = None,
            output_layer_activation: Literal["linear", "relu", "sigmoid", "tanh", "softmax"] = "linear",
            loss_function: Literal["mse", "bce", "cce"] = "mse",
            weight_initialization: Literal["zero", "uniform", "normal"] = "zero",
            lower_bound: None | float = None,
            upper_bound: None | float = None,
            mean: None | float = None,
            variance: None | float = None,
            random_seed: int = 42,
            batch_size: int = 10,
            epochs: int = 5,
            optimizer: Literal["adams", "gd"] = "gd",
            learning_rate: float = 0.1,
            momentum_gain: None | float = None,
            rms_gain: None | float = None,
            l1_strength: float = 0,
            l2_strength: float = 0,
            verbose: bool = False,
        ):
        '''
        Initializes a FFNN with the specified parameters

        Args:
            hidden_layer_sizes (list[int]): array of integers specifying how many perceptrons are in each hidden layer\n
            hidden_layer_activations (None | list[Literal["linear", "relu", "sigmoid", "tanh", "softmax"]]): array of string specifying activation function of each hidden layer\n
            output_layer_activation (Literal["linear", "relu", "sigmoid", "tanh", "softmax"]): string specifying activation function of output layer\n
            loss_function ("mse" | "bce" | "cce"): string specifying the loss function used for training\n
            weight_initialization ("zero" | "uniform" | "normal"): string specifying the weight initialization method of the model\n
            lower_bound (None | float): lower bound of random weight when using uniform distribution\n
            upper_bound (None | float): upper bound of random weight when using uniform distribution\n
            mean (None | float): mean of random weights when using normal distribution\n
            variance (None | float): variance of random weights when using normal distribution\n
            random_seed (int): integer used as a random seed for random weights or sampling order during training\n
            batch_size (int): size of single batch of sample data used during training for each epoch\n
            epochs (int): how many batches should the model use during training\n
            optimizer ("gd" | "adams"): the preceptron optimizer function used\n
            learning_rate (None | float): learning rate of the model when using gradient descent optimizer\n
            momentum_gain (None | float): value of beta 1 parameter when using adams optimizer\n
            rms_gain (None | float): value of beta 2 parameter when using adams optimizer\n
            l1_strength (float): strength of L1 regularization during model training\n
            l2_strength (float): strength of L2 regularization during model training\n
            verbose (bool): whether the model should log its progress during training or fitting\n
        '''

        if ((hidden_layer_sizes is None) != (hidden_layer_activations is None)):
            raise ValueError("hidden layer sizes and activations must both be provided or both be None")

        if hidden_layer_sizes is not None and hidden_layer_activations is not None and len(hidden_layer_sizes) != len(hidden_layer_activations):
            raise ValueError("hidden layer sizes count must match hidden layer activations count")

        self._hidden_layer_count = len(hidden_layer_sizes) if hidden_layer_sizes is not None else None
        self._hidden_layer_sizes = hidden_layer_sizes

        if hidden_layer_activations and any(activation not in ["linear", "relu", "sigmoid", "tanh", "softmax"] for activation in hidden_layer_activations):
            raise ValueError("unknown activation function in hidden layers")
        
        self._hidden_layer_activations = hidden_layer_activations

        if (output_layer_activation not in ["linear", "relu", "sigmoid", "tanh", "softmax"]):
            raise ValueError("unknown output layer activation function")

        self._output_layer_activation = output_layer_activation

        if (loss_function not in ["mse", "bce", "cce"]):
            raise ValueError("unknown loss function")

        self._loss_function = loss_function

        if (weight_initialization not in ["zero", "uniform", "normal"]):
            raise ValueError("unknown weight initialization method")

        self._weight_initialization = weight_initialization

        if (weight_initialization == "uniform" and (lower_bound is None or upper_bound is None)):
            raise ValueError("lower_bound and upper_bound are required for uniform initialization")

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        if (weight_initialization == "normal" and (mean is None or variance is None)):
            raise ValueError("mean and variance are required for normal initialization")

        self._mean = mean
        self._variance = variance

        self._random_seed = random_seed
        self._batch_size = batch_size
        self._epochs = epochs

        self._optimizer = optimizer

        if (optimizer != "gd" and optimizer != "adams"):
            raise ValueError("unknown optimizer")

        if learning_rate is None:
            raise ValueError("learning_rate must be set")
        
        self._learning_rate = learning_rate

        if (optimizer == "adams" and (momentum_gain is None or rms_gain is None)):
            raise ValueError("momentum_gain and rms_gain must be set when using adams optimizer")

        self._momentum_gain = momentum_gain
        self._rms_gain = rms_gain

        self._l1_strength = l1_strength
        self._l2_strength = l2_strength

        self._verbose = verbose


    def _build_computation_graph(self):
        activations = list(self._hidden_layer_activations or [])
        activations.append(self._output_layer_activation)

        layer_sizes = list(self._hidden_layer_sizes or [])
        layer_sizes.append(self._label_count)

        self._bias_broadcasts: list[ADVBroadcastTo] = []
        self._in_matrix = ADVMatrix()

        A = ADVMatTrans(self._in_matrix)

        for fn, sz, W, b in zip(activations, layer_sizes, self._weights, self._bias):
            B_shape = (sz, self._batch_size) if self._batch_size else None

            WA = ADVMatMul(W, A)
            B = ADVBroadcastTo(b, B_shape)
            Z = ADVMatAdd(WA, B)

            self._bias_broadcasts.append(B)

            match fn:
                case "linear":
                    A = Z
                case "relu":
                    A = ADVReLU(Z)
                case "sigmoid":
                    A = ADVSigmoid(Z)
                case "tanh":
                    A = ADVTanh(Z)
                case "softmax":
                    T = ADVMatTrans(Z)
                    S = ADVSoftmax(T)
                    A = ADVMatTrans(S)
                case _:
                    raise ValueError("unknown activation function found in model")

        self._out_matrix = ADVMatTrans(A)

        match self._loss_function:
            case "mse":
                self._loss = ADVMeanSquaredError(self._out_matrix)
            case "bce":
                self._loss = ADVBinaryCrossEntropy(self._out_matrix)
            case "cce":
                self._loss = ADVCategoricalCrossEntropy(self._out_matrix)

            
    def save(self, file_path: str):
        '''
        Saves the current model to a file

        Args:
            file_path (str): path to the save file
        '''

        if not hasattr(self, "_weights") or not hasattr(self, "_bias"):
            raise RuntimeError("model has not been trained; fit the model before saving")

        labels = None
        if not getattr(self, "_one_hot_encoded", False):
            labels = self._labels.tolist() if hasattr(self, "_labels") else None

        weights = [w.value for w in self._weights]
        w_grads = [w.gradient for w in self._weights]
        biases = [b.value for b in self._bias]
        b_grads = [b.gradient for b in self._bias]

        payload = {
            "feature_count": int(self._feature_count),
            "label_count": int(self._label_count),
            "hidden_layers": [{"size": l, "activation": a} for l, a in zip(self._hidden_layer_sizes or [], self._hidden_layer_activations or [])],
            "final_activation": self._output_layer_activation,
            "one_hot_encoded": bool(self._one_hot_encoded),
            "labels": labels,
            "weight_count": len(weights),
            "bias_count": len(biases),
        }
    
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.array(json.dumps(payload)),
        }
    
        for i, weight in enumerate(weights):
            arrays[f"weight_{i}"] = np.asarray(weight)

        for i, grad in enumerate(w_grads):
            arrays[f"w_grad_{i}"] = np.asarray(grad)
    
        for i, bias in enumerate(biases):
            arrays[f"bias_{i}"] = np.asarray(bias)
        
        for i, grad in enumerate(b_grads):
            arrays[f"b_grad_{i}"] = np.asarray(grad)
    
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(file_path, **arrays)


    @classmethod
    def load(cls, file_path: str):
        '''
        Saves the current model to a file

        Args:
            file_path (str): path to the save file
        
        Returns:
            model (FFNN): the FFNN model stored
        '''

        with np.load(file_path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata_json"].item())
    
            weight_count = int(metadata["weight_count"])
            bias_count = int(metadata["bias_count"])
    
            weights = [data[f"weight_{i}"].copy() for i in range(weight_count)]
            w_grads = [data[f"w_grad_{i}"].copy() for i in range(weight_count)]
            biases = [data[f"bias_{i}"].copy() for i in range(bias_count)]
            b_grads = [data[f"b_grad_{i}"].copy() for i in range(bias_count)]
        
        hidden_layers = metadata["hidden_layers"]
        hidden_layer_sizes = [l["size"] for l in hidden_layers]
        hidden_layer_activations = [l["activation"] for l in hidden_layers]

        model = cls(hidden_layer_sizes, hidden_layer_activations, metadata["final_activation"])

        model._feature_count = int(metadata["feature_count"])
        model._label_count = int(metadata["label_count"])
        model._one_hot_encoded = bool(metadata["one_hot_encoded"])

        labels = metadata.get("labels")
        model._labels = np.array(labels) if labels is not None else None

        if not model._one_hot_encoded and model._labels is None:
            raise ValueError("loaded model requires class labels but labels are missing in metadata")
        
        model._weights = [ADVMatrix(w) for w in weights]
        model._bias = [ADVMatrix(b) for b in biases]

        for w, g in zip(model._weights, w_grads):
            w.gradient = g

        for b, g in zip(model._bias, b_grads):
            b.gradient = g

        model._build_computation_graph()

        return model

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Fits the model to the given training data

        Args:
            X (ndarray): training data features
            y (ndarray): training data labels
        '''
        
        X = np.asarray(X)
        y = np.asarray(y)

        # Set RNG
        rng = np.random.default_rng(seed=self._random_seed)

        # Find output layer size
        self._one_hot_encoded = y.ndim == 2

        if not self._one_hot_encoded:
            # If the labels are not one hot encoded, we need to keep track of the labels
            self._labels = np.unique(y)
            self._label_count = len(self._labels)
            y_ohe = (y[:, None] == self._labels).astype(float)
        else:
            self._label_count = y.shape[1]
            y_ohe = y
        
        # Find input layer size
        self._feature_count = X.shape[1]

        # Determine the weight and bias init func
        match self._weight_initialization:
            case "zero":
                weight_init_func = lambda shape: ADVMatrix(np.zeros(*shape))
            case "uniform":
                weight_init_func = lambda shape: ADVMatrix(rng.uniform(low=self._lower_bound, high=self._upper_bound, size=shape))
            case "normal":
                weight_init_func = lambda shape: ADVMatrix(rng.normal(loc=self._mean, scale=self._variance ** 0.5, size=shape))
            case _:
                raise ValueError("unknown weight initialization method stored in model")

        # Initialize all weights and biases to 0
        if not self._hidden_layer_count:
            self._weights = [weight_init_func((self._label_count, self._feature_count))]
            self._bias = [weight_init_func((self._label_count, 1))]
        else:
            self._weights = [
                weight_init_func((self._hidden_layer_sizes[0], self._feature_count))
            ] + [
                weight_init_func((self._hidden_layer_sizes[i], self._hidden_layer_sizes[i - 1]))
                for i in range(1, self._hidden_layer_count)
            ] + [
                weight_init_func((self._label_count, self._hidden_layer_sizes[-1]))
            ]

            self._bias = [
                weight_init_func((self._hidden_layer_sizes[i], 1))
                for i in range(0, self._hidden_layer_count)
            ] + [weight_init_func((self._label_count, 1))]
        
        # Initialize optimizers
        match self._optimizer:
            case "gd":
                optimizer_factory = lambda w: GradientDescentOptimizer(w, self._learning_rate, self._l1_strength, self._l2_strength)
            case "adams":
                optimizer_factory = lambda w: AdamOptimizer(w, self._learning_rate, self._momentum_gain, self._rms_gain, self._l1_strength, self._l2_strength)
            case _:
                raise ValueError("unknown optimizer stored in model")
        
        weight_optimizers = [optimizer_factory(w.value) for w in self._weights]
        bias_optimizers = [optimizer_factory(b.value) for b in self._bias]

        # Build computation graph
        self._build_computation_graph()

        # Scramble the training data
        needed_samples = self._batch_size * self._epochs

        training_data = np.concatenate((X, y_ohe), axis=1)
        training_order = rng.permutation(training_data)

        while training_order.shape[0] < needed_samples:
            training_order = np.concatenate((training_order, rng.permutation(training_data)), axis=0)
        
        training_order = training_order[:needed_samples, :].reshape((self._epochs, self._batch_size, training_data.shape[1]))

        # Train on every batch
        for batch in training_order:
            self._in_matrix.value = batch[:, :-self._label_count]
            self._loss.targets = batch[:, -self._label_count:]
            self._loss.calculate_value()
            self._loss.calculate_backward_gradients()

            for w, b, wo, bo in zip(self._weights, self._bias, weight_optimizers, bias_optimizers):
                w.value = wo.optimize(w.gradient)
                b.value = bo.optimize(b.gradient)
                
    
    def predict(self, X: np.ndarray):
        '''
        Predict class labels for samples in X

        Args:
            X (ndarray): samples
        
        Returns:
            y_pred (ndarray): prediction results
        '''

        sample_size = X.shape[0]

        layer_sizes = list(self._hidden_layer_sizes or [])
        layer_sizes.append(self._label_count)

        for B, sz in zip(self._bias_broadcasts, layer_sizes):
            B.target_shape = (sz, sample_size)
        
        self._in_matrix.value = X
        y_pred_ohe = self._out_matrix.calculate_value()

        if self._one_hot_encoded:
            y_pred = y_pred_ohe
        else:
            y_pred = self._labels[y_pred_ohe.argmax(axis=1)]
        
        return y_pred
