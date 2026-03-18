import json
from pathlib import Path
from typing import Literal
from optimizer import *
from autodiff import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class FFNN:
    "Feedforward Neural Network implementation."
    
    def __init__(
            self,
            hidden_layer_sizes: None | list[int] = None,
            hidden_layer_activations: None | list[Literal["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"]] = None,
            output_layer_activation: Literal["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"] = "linear",
            loss_function: Literal["mse", "bce", "cce"] = "mse",
            weight_initialization: Literal["zero", "uniform", "normal", "xavier", "he"] = "zero",
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
            rmsnorm: bool = False,
            verbose: bool = False,
        ):
        '''
        Initializes a FFNN with the specified parameters

        Args:
            hidden_layer_sizes (list[int]): array of integers specifying how many perceptrons are in each hidden layer\n
            hidden_layer_activations (None | list[Literal["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"]]): array of string specifying activation function of each hidden layer\n
            output_layer_activation (Literal["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"]): string specifying activation function of output layer\n
            loss_function ("mse" | "bce" | "cce"): string specifying the loss function used for training\n
            weight_initialization ("zero" | "uniform" | "normal" | "xavier" | "he"): string specifying the weight initialization method of the model\n
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
            rmsnorm (bool): whether the model uses rmsnorm regularization\n
            verbose (bool): whether the model should log its progress during training or fitting\n
        '''

        if ((hidden_layer_sizes is None) != (hidden_layer_activations is None)):
            raise ValueError("hidden layer sizes and activations must both be provided or both be None")

        if hidden_layer_sizes is not None and hidden_layer_activations is not None and len(hidden_layer_sizes) != len(hidden_layer_activations):
            raise ValueError("hidden layer sizes count must match hidden layer activations count")

        self._hidden_layer_count = len(hidden_layer_sizes) if hidden_layer_sizes is not None else None
        self._hidden_layer_sizes = hidden_layer_sizes

        if hidden_layer_activations and any(activation not in ["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"] for activation in hidden_layer_activations):
            raise ValueError("unknown activation function in hidden layers")
        
        self._hidden_layer_activations = hidden_layer_activations

        if (output_layer_activation not in ["linear", "relu", "sigmoid", "tanh", "softmax", "sign", "softplus"]):
            raise ValueError("unknown output layer activation function")

        self._output_layer_activation = output_layer_activation

        if (loss_function not in ["mse", "bce", "cce"]):
            raise ValueError("unknown loss function")

        self._loss_function = loss_function

        if (weight_initialization not in ["zero", "uniform", "normal", "xavier", "he"]):
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

        self._rmsnorm = rmsnorm

        self._verbose = verbose


    def _build_computation_graph(self):
        activations = list(self._hidden_layer_activations or [])
        activations.append(self._output_layer_activation)

        layer_sizes = list(self._hidden_layer_sizes or [])
        layer_sizes.append(self._label_count)

        self._bias_broadcasts: list[ADVBroadcastTo] = []

        if self._rmsnorm:
            self._rmsnorm_broadcasts: list[ADVBroadcastTo] = []

        self._in_matrix = ADVMatrix()

        A = ADVMatTrans(self._in_matrix)

        for i, (fn, W, b) in enumerate(zip(activations, self._weights, self._bias)):
            if self._rmsnorm:
                Wr = ADVBroadcastTo(self._rms_weights[i])
                WrT = ADVMatTrans(Wr)
                A = ADVMatTrans(A)
                A = ADVRMSNorm(A, WrT)
                A = ADVMatTrans(A)

                self._rmsnorm_broadcasts.append(Wr)

            WA = ADVMatMul(W, A)
            B = ADVBroadcastTo(b, None)
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
                case "sign":
                    A = ADVSign(Z)
                case "softplus":
                    A = ADVSoftPlus(Z)
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
            "loss_function": self._loss_function,
            "weight_initialization": self._weight_initialization,
            "lower_bound": self._lower_bound,
            "upper_bound": self._upper_bound,
            "mean": self._mean,
            "variance": self._variance,
            "random_seed": self._random_seed,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "optimizer": self._optimizer,
            "learning_rate": self._learning_rate,
            "momentum_gain": self._momentum_gain,
            "rms_gain": self._rms_gain,
            "l1_strength": self._l1_strength,
            "l2_strength": self._l2_strength,
            "rmsnorm": self._rmsnorm,
            "verbose": self._verbose,
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
    
        if self._rmsnorm:
            for i, rms in enumerate(self._rms_weights):
                arrays[f"rms_weight_{i}"] = np.asarray(rms.value)
                arrays[f"rms_grad_{i}"] = np.asarray(rms.gradient)
    
        if hasattr(self, '_loss_history'):
            epochs_trained = len(self._loss_history)
            payload["epochs_trained"] = epochs_trained
            arrays["loss_history"] = np.array(self._loss_history)
            if hasattr(self, '_validation_loss_history') and len(self._validation_loss_history) > 0:
                arrays["validation_loss_history"] = np.array(self._validation_loss_history)
            for epoch in range(epochs_trained):
                for layer_idx, w in enumerate(self._weights_history[epoch]):
                    arrays[f"weights_history_{epoch}_{layer_idx}"] = np.asarray(w)
                for layer_idx, b in enumerate(self._biases_history[epoch]):
                    arrays[f"biases_history_{epoch}_{layer_idx}"] = np.asarray(b)
                for layer_idx, wg in enumerate(self._weights_grad_history[epoch]):
                    arrays[f"weights_grad_history_{epoch}_{layer_idx}"] = np.asarray(wg)
                for layer_idx, bg in enumerate(self._biases_grad_history[epoch]):
                    arrays[f"biases_history_{epoch}_{layer_idx}"] = np.asarray(bg)
    
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(file_path, **arrays)


    @classmethod
    def load(cls, file_path: str):
        '''
        Loads a FFNN model from a file

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
        hidden_layer_sizes = [l["size"] for l in hidden_layers] if hidden_layers else None
        hidden_layer_activations = [l["activation"] for l in hidden_layers] if hidden_layers else None

        model = cls(
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_activations=hidden_layer_activations,
            output_layer_activation=metadata["final_activation"],
            loss_function=metadata["loss_function"],
            weight_initialization=metadata["weight_initialization"],
            lower_bound=metadata.get("lower_bound"),
            upper_bound=metadata.get("upper_bound"),
            mean=metadata.get("mean"),
            variance=metadata.get("variance"),
            random_seed=metadata["random_seed"],
            batch_size=metadata["batch_size"],
            epochs=metadata["epochs"],
            optimizer=metadata["optimizer"],
            learning_rate=metadata["learning_rate"],
            momentum_gain=metadata.get("momentum_gain"),
            rms_gain=metadata.get("rms_gain"),
            l1_strength=metadata["l1_strength"],
            l2_strength=metadata["l2_strength"],
            rmsnorm=metadata["rmsnorm"],
            verbose=metadata["verbose"],
        )

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

        if metadata.get("rmsnorm", False):
            rms_weights = [data[f"rms_weight_{i}"].copy() for i in range(weight_count)]
            rms_grads = [data[f"rms_grad_{i}"].copy() for i in range(weight_count)]
            model._rms_weights = [ADVMatrix(r) for r in rms_weights]
            for r, g in zip(model._rms_weights, rms_grads):
                r.gradient = g

        model._build_computation_graph()

        if metadata.get("epochs_trained", 0) > 0:
            epochs_trained = metadata["epochs_trained"]
            model._loss_history = data["loss_history"].tolist()
            if "validation_loss_history" in data:
                model._validation_loss_history = data["validation_loss_history"].tolist()
            else:
                model._validation_loss_history = []
            model._weights_history = []
            model._biases_history = []
            model._weights_grad_history = []
            model._biases_grad_history = []
            for epoch in range(epochs_trained):
                w_hist = []
                b_hist = []
                wg_hist = []
                bg_hist = []
                for layer in range(len(model._weights)):
                    w_hist.append(data[f"weights_history_{epoch}_{layer}"].copy())
                    b_hist.append(data[f"biases_history_{epoch}_{layer}"].copy())
                    wg_hist.append(data[f"weights_grad_history_{epoch}_{layer}"].copy())
                    bg_hist.append(data[f"biases_grad_history_{epoch}_{layer}"].copy())
                model._weights_history.append(w_hist)
                model._biases_history.append(b_hist)
                model._weights_grad_history.append(wg_hist)
                model._biases_grad_history.append(bg_hist)

        return model

    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        '''
        Fits the model to the given training data

        Args:
            X (ndarray): training data features
            y (ndarray): training data labels
            X_val (ndarray | None): validation data features (optional)
            y_val (ndarray | None): validation data labels (optional)
        '''
        
        X = np.asarray(X)
        y = np.asarray(y)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        # Set RNG
        rng = np.random.default_rng(seed=self._random_seed)

        # Find output layer size
        self._one_hot_encoded = y.ndim == 2

        if not self._one_hot_encoded:
            # If the labels are not one hot encoded, we need to keep track of the labels
            self._labels = np.unique(y)
            self._label_count = len(self._labels)
            y_ohe = (y[:, None] == self._labels).astype(float)

            if X_val is not None:
                y_val_ohe = (y_val[:, None] == self._labels).astype(float)
        else:
            self._label_count = y.shape[1]
            y_ohe = y

            if X_val is not None:
                y_val_ohe = y_val
        
        # Find input layer size
        self._feature_count = X.shape[1]

        if self._verbose:
            print(f"Starting training with {X.shape[0]} samples, {self._feature_count} features, {self._label_count} labels")

        # Determine the weight and bias init func
        match self._weight_initialization:
            case "zero":
                weight_init_func = lambda shape: ADVMatrix(np.zeros(shape))
            case "uniform":
                weight_init_func = lambda shape: ADVMatrix(rng.uniform(low=self._lower_bound, high=self._upper_bound, size=shape))
            case "normal":
                weight_init_func = lambda shape: ADVMatrix(rng.normal(loc=self._mean, scale=self._variance ** 0.5, size=shape))
            case "xavier":
                weight_init_func = lambda shape: ADVMatrix(rng.normal(loc=0, scale=(2 / (shape[1] + shape[0])) ** 0.5, size=shape))
            case "he":
                weight_init_func = lambda shape: ADVMatrix(rng.normal(loc=0, scale=(2 / shape[1]) ** 0.5, size=shape))
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
        
        if self._verbose:
            total_params = sum(w.value.size for w in self._weights) + sum(b.value.size for b in self._bias)
            print(f"Weights and biases initialized with {total_params} parameters")
        
        # Initialize RMSNorm weights
        if self._rmsnorm:
            self._rms_weights = [ADVMatrix(np.ones(shape=(sz, 1))) for sz in [self._feature_count] + (self._hidden_layer_sizes or [])]
        
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

        if self._rmsnorm:
            rmsnorm_optimizers = [optimizer_factory(r.value) for r in self._rms_weights]

        if self._verbose:
            print("Optimizers initialized")

        # Build computation graph
        self._build_computation_graph()

        if self._verbose:
            print("Computation graph built")

        training_data = np.concatenate((X, y_ohe), axis=1)
        
        n_samples = training_data.shape[0]

        if self._verbose:
            print(f"Training: {self._epochs} epochs, batch_size={self._batch_size}, n_samples={n_samples}")

        # Initialize history storage
        self._weights_history = []
        self._biases_history = []
        self._weights_grad_history = []
        self._biases_grad_history = []
        self._loss_history = []
        self._validation_loss_history = []

        # Train: each epoch = one full pass over the dataset, divided into mini-batches.
        # Weights are updated after each mini-batch (standard mini-batch SGD).
        # Loss recorded per epoch is the average loss over all mini-batches in that epoch.
        for epoch_idx in range(self._epochs):
            if self._verbose:
                print(f"Epoch {epoch_idx + 1}/{self._epochs}")

            # Shuffle dataset at the start of each epoch
            shuffled = rng.permutation(training_data)

            # Split into mini-batches; last batch may be smaller (drop if < 1 sample)
            n_batches = max(1, n_samples // self._batch_size)
            batches = np.array_split(shuffled, n_batches)

            epoch_loss = 0.0
            epoch_last_w_grad = None
            epoch_last_b_grad = None

            for batch_idx, batch in enumerate(batches):
                if len(batch) == 0:
                    continue

                # Dynamically update bias broadcast target shape for this batch size
                actual_batch_size = len(batch)
                layer_sizes = list(self._hidden_layer_sizes or [])
                layer_sizes.append(self._label_count)

                if self._verbose:
                    print(f"    Batch {batch_idx}/{len(batches) - 1}")
                    print(f"        Batch Size: {actual_batch_size}")
                
                for B, sz in zip(self._bias_broadcasts, layer_sizes):
                    B.target_shape = (sz, actual_batch_size)
                
                if self._rmsnorm:
                    for B, sz in zip(self._rmsnorm_broadcasts, [self._feature_count] + (self._hidden_layer_sizes or [])):
                        B.target_shape = (sz, actual_batch_size)

                # Forward pass
                self._loss.clear_gradients()
                self._in_matrix.value = batch[:, :-self._label_count]
                self._loss.targets = batch[:, -self._label_count:]
                self._loss.calculate_value()

                if self._verbose:
                    print(f"        Batch Loss: {self._loss.value}")

                epoch_loss += self._loss.value

                # Backward pass
                self._loss.calculate_backward_gradients()

                # Weight update after each mini-batch
                for i, (w, b, wo, bo) in enumerate(zip(self._weights, self._bias, weight_optimizers, bias_optimizers)):
                    w.value = wo.optimize(w.gradient)
                    b.value = bo.optimize(b.gradient)

                    if self._rmsnorm:
                        r = self._rms_weights[i]
                        r.value = rmsnorm_optimizers[i].optimize(r.gradient)

                epoch_last_w_grad = [w.gradient.copy() for w in self._weights]
                epoch_last_b_grad = [b.gradient.copy() for b in self._bias]

            avg_epoch_loss = epoch_loss / n_batches
            self._loss_history.append(avg_epoch_loss)

            # Calculate validation loss
            if X_val is not None:
                self._in_matrix.value = X_val
                self._loss.targets = y_val_ohe

                for B, sz in zip(self._bias_broadcasts, layer_sizes):
                    B.target_shape = (sz, X_val.shape[0])
                
                if self._rmsnorm:
                    for B, sz in zip(self._rmsnorm_broadcasts, [self._feature_count] + (self._hidden_layer_sizes or [])):
                        B.target_shape = (sz, X_val.shape[0])

                validation_loss = self._loss.calculate_value()
                self._validation_loss_history.append(validation_loss)

            if self._verbose:
                print(f"  Epoch avg loss : {avg_epoch_loss:.6f}")
                if validation_loss:
                    print(f"  Validation loss: {validation_loss:.6f}")

            # Store snapshots after each epoch
            self._weights_history.append([w.value.copy() for w in self._weights])
            self._biases_history.append([b.value.copy() for b in self._bias])
            self._weights_grad_history.append(epoch_last_w_grad if epoch_last_w_grad else [np.zeros_like(w.value) for w in self._weights])
            self._biases_grad_history.append(epoch_last_b_grad if epoch_last_b_grad else [np.zeros_like(b.value) for b in self._bias])

        if self._verbose:
            print("Training completed")
            
    
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
        
        if self._rmsnorm:
            for B, sz in zip(self._rmsnorm_broadcasts, [self._feature_count] + (self._hidden_layer_sizes or [])):
                B.target_shape = (sz, sample_size)
        
        self._in_matrix.value = X
        y_pred_ohe = self._out_matrix.calculate_value()

        if self._one_hot_encoded:
            y_pred = y_pred_ohe
        else:
            y_pred = self._labels[y_pred_ohe.argmax(axis=1)]
        
        return y_pred


    def show_weight_distribution(self, layers: list[int], bins: int = 50):
        '''
        Displays a 3D distribution surface of weights for specified layers across epochs.

        Args:
            layers (list[int]): list of layer indices to observe
            bins (int): number of histogram bins
        '''
        if not hasattr(self, '_weights_history'):
            raise ValueError("No training history available. Fit the model first.")

        epochs = len(self._weights_history)
        epoch_nums = np.arange(1, epochs + 1)

        fig = plt.figure(figsize=(10, 5 * len(layers)))
        axes = [fig.add_subplot(len(layers), 1, i + 1, projection='3d') for i in range(len(layers))]

        # Precompute histograms for each layer and each epoch using a consistent binning
        hist_data = {}
        bins_edges = {}
        for layer_idx in layers:
            all_weights = np.concatenate([self._weights_history[e][layer_idx].flatten() for e in range(epochs)])
            edges = np.histogram_bin_edges(all_weights, bins=bins)
            bins_edges[layer_idx] = edges
            hist_data[layer_idx] = np.stack(
                [np.histogram(self._weights_history[e][layer_idx].flatten(), bins=edges)[0] for e in range(epochs)]
            )

        for i, layer_idx in enumerate(layers):
            ax = axes[i]
            edges = bins_edges[layer_idx]
            centers = (edges[:-1] + edges[1:]) / 2

            X, Y = np.meshgrid(centers, epoch_nums)
            Z = hist_data[layer_idx]

            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(f'Layer {layer_idx} Weights Distribution over Epochs')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Epoch')
            ax.set_zlabel('Frequency')

        plt.tight_layout()
        plt.show()


    def show_gradient_distribution(self, layers: list[int], bins: int = 50):
        '''
        Displays a 3D distribution surface of gradients for specified layers across epochs.

        Args:
            layers (list[int]): list of layer indices to observe
            bins (int): number of histogram bins
        '''
        if not hasattr(self, '_weights_grad_history'):
            raise ValueError("No training history available. Fit the model first.")

        epochs = len(self._weights_grad_history)
        epoch_nums = np.arange(1, epochs + 1)

        fig = plt.figure(figsize=(10, 5 * len(layers)))
        axes = [fig.add_subplot(len(layers), 1, i + 1, projection='3d') for i in range(len(layers))]

        # Precompute histograms for each layer and each epoch using a consistent binning
        hist_data = {}
        bins_edges = {}
        for layer_idx in layers:
            all_grads = np.concatenate([self._weights_grad_history[e][layer_idx].flatten() for e in range(epochs)])
            edges = np.histogram_bin_edges(all_grads, bins=bins)
            bins_edges[layer_idx] = edges
            hist_data[layer_idx] = np.stack(
                [np.histogram(self._weights_grad_history[e][layer_idx].flatten(), bins=edges)[0] for e in range(epochs)]
            )

        for i, layer_idx in enumerate(layers):
            ax = axes[i]
            edges = bins_edges[layer_idx]
            centers = (edges[:-1] + edges[1:]) / 2

            X, Y = np.meshgrid(centers, epoch_nums)
            Z = hist_data[layer_idx]

            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(f'Layer {layer_idx} Gradients Distribution over Epochs')
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Epoch')
            ax.set_zlabel('Frequency')

        plt.tight_layout()
        plt.show()


    def plot_loss(self):
        '''
        Plots the loss over epochs.
        '''
        if not hasattr(self, '_loss_history'):
            raise ValueError("No training history available. Fit the model first.")

        epochs = range(1, len(self._loss_history) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self._loss_history, label='Training Loss')
        if hasattr(self, '_validation_loss_history') and len(self._validation_loss_history) > 0:
            plt.plot(epochs, self._validation_loss_history, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
