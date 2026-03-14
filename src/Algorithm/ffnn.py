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
            verbose (bool): whether the model should log its progress during training or fitting\n
        '''

        if (((hidden_layer_sizes == None) != (hidden_layer_activations == None)) or (len(hidden_layer_sizes) != len(hidden_layer_activations))):
            '''TODO: Error on mismatching hidden layer count'''

        self._hidden_layer_count = len(hidden_layer_sizes) if hidden_layer_sizes != None else None
        self._hidden_layer_sizes = hidden_layer_sizes

        if (hidden_layer_activations and (activation not in ["linear", "relu", "sigmoid", "tanh", "softmax"] for activation in hidden_layer_activations)):
            '''TODO: Error on unrecognized activation function'''
        
        self._hidden_layer_activations = hidden_layer_activations

        if (output_layer_activation not in ["linear", "relu", "sigmoid", "tanh", "softmax"]):
            '''TODO: Error on unrecognized activation function'''

        self._output_layer_activation = output_layer_activation

        if (loss_function not in ["mse", "bce", "cce"]):
            '''TODO: Error on unrecognized loss function'''

        self._loss_function = loss_function

        if (weight_initialization not in ["zero" | "uniform" | "normal"]):
            '''TODO: Error on unrecognized weight initialization method'''

        self._weight_initialization = weight_initialization

        if (weight_initialization == "uniform" and (lower_bound == None or upper_bound == None)):
            '''TODO: Error on lower or upper bound not set with uniform weight initialization'''

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        if (weight_initialization == "normal" and (mean == None or variance == None)):
            '''TODO: Error on mean or variance not set with normal weight initialization'''

        self._mean = mean
        self._variance = variance

        self._random_seed = random_seed
        self._batch_size = batch_size
        self._epochs = epochs

        self._optimizer = optimizer

        if (optimizer != "gd" and optimizer != "adams"):
            '''TODO: Error on unrecognized optimizer'''

        if not learning_rate:
            '''TODO: learning rate not set'''
        
        self._learning_rate = learning_rate

        if (optimizer == "adams" and (momentum_gain == None or rms_gain == None)):
            '''TODO: momentum gain or rms gain not set with adams optimizer'''

        self._momentum_gain = momentum_gain
        self._rms_gain = rms_gain

        self._verbose = verbose

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Fits the model to the given training data

        Args:
            X (ndarray): training data features
            y (ndarray): training data labels
        '''
        
        # Set RNG
        rng = np.random.default_rng(seed=self._random_seed)

        # Find output layer size
        self._one_hot_encoded = y.ndim == 2

        if not self._one_hot_encoded:
            # If the labels are not one hot encoded, we need to keep track of the labels
            self._labels = np.unique(y)
            self._label_count = len(self._labels)
        else:
            self._label_count = y.shape[1]
        
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
                optimizer_factory = lambda w: GradientDescentOptimizer(w, self._learning_rate)
            case "adam":
                optimizer_factory = lambda w: AdamOptimizer(w, self._learning_rate, self._momentum_gain, self._rms_gain)
            case _:
                raise ValueError("unknown optimizer stored in model")
        
        weight_optimizers = [optimizer_factory(w.value) for w in self._weights]
        bias_optimizers = [optimizer_factory(b.value) for b in self._bias]

        # Build computation graph
        activations = (self._hidden_layer_activations or [])
        activations.append(self._output_layer_activation)

        layer_sizes = (self._hidden_layer_sizes or [])
        layer_sizes.append(self._label_count)

        self._bias_broadcasts: list[ADVBroadcastTo] = []
        self._in_matrix = ADVMatrix()

        A = ADVMatTrans(self._in_matrix)

        for fn, sz, W, b in zip(activations, layer_sizes, self._weights, self._bias):
            B_shape = (sz, self._batch_size)
            
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
            case _:
                raise ValueError("unknown loss function found in model")

        # Scramble the training data
        needed_samples = self._batch_size * self._epochs

        training_data = np.concatenate(X, y, axis=1)
        training_order = rng.permutation(training_data)

        while training_order.shape[0] < needed_samples:
            training_order = np.concatenate(training_order, rng.permutation(training_data))
        
        training_order = training_order[:needed_samples, :].reshape((self._batch_size, *training_data.shape))

        # Train on every batch
        for batch in training_order:
            self._in_matrix.value = batch
            self._loss.calculate_value()
            self._loss.calculate_backward_gradients()

            for w, b, wo, bo in zip(self._weights, self._bias, weight_optimizers, bias_optimizers):
                w.value = wo.optimize(w.gradient)
                b.value = bo.optimize(b.gradient)
                
    
    def predict(self, X):
        '''
        Predict class labels for samples in X

        Args:
            X (ndarray): samples
        
        Returns:
            y_pred (ndarray): prediction results
        '''
    
