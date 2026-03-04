from typing import Literal
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
            optimization: None | Literal["adams"] = None,
            learning_rate: float = 0.1,
            beta_1: None | float = None,
            beta_2: None | float = None,
            epsilon: None | float = None,
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
            optimization (None | "adams"): None for no optimization or "adams" for adams optimization\n
            learning_rate (None | float): learning rate of the model when using no optimization\n
            beta_1 (None | float): value of beta 1 parameter when using adams optimization\n
            beta_2 (None | float): value of beta 2 parameter when using adams optimization\n
            epsilon (None | float): value of epsilon parameter when using adams optimization\n
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

        if (optimization and optimization != "adams"):
            '''TODO: Error on unrecognized optimization method'''

        self._optimization = optimization

        if (not optimization and not learning_rate):
            '''TODO: learning rate not set with no optimization method'''

        self._learning_rate = learning_rate

        if (optimization == "admas" and (beta_1 == None or beta_2 == None or epsilon == None)):
            '''TODO: beta 1, beta 2, or epsilon not set with adams optimization method'''

        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon

        self._verbose = verbose

    
    def fit(self, X, y):
        '''
        Fits the model to the given training data

        Args:
            X (ndarray): training data features
            y (ndarray): training data labels
        '''

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
                weight_init_func = lambda shape: np.zeros(*shape)
            case "uniform":
                weight_init_func = lambda shape: np.random.rand(*shape) * (self._upper_bound - self._lower_bound) + self._lower_bound
            case "normal":
                weight_init_func = lambda shape: np.random.randn(*shape) * self._variance ** 0.5 + self._mean
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

    
    def predict(self, X):
        '''
        Predict class labels for samples in X

        Args:
            X (ndarray): samples
        
        Returns:
            y_pred (ndarray): prediction results
        '''