from abc import ABC, abstractmethod
import numpy as np

class AutoDifferentiableValue(ABC):
    '''Abstract base class for all values that can be auto differentiable as part of a computation graph'''

    def __init__(self):
        self.gradient: None | np.ndarray = None
        self.value: None | np.ndarray | float = None
    
    @abstractmethod
    def clear_gradients(self):
        raise NotImplementedError()
    
    @abstractmethod
    def calculate_value(self) -> None | np.ndarray | float:
        raise NotImplementedError()

    def calculate_backward_gradients(self, _upstream: np.ndarray | None = None) -> np.ndarray:
        if _upstream == None:
            self.gradient = np.ones_like(self.value)
            return self.gradient.copy()
        elif self.gradient is None:
            self.gradient = _upstream.copy()
            return _upstream
        else:
            self.gradient += _upstream
            return _upstream


class ADVMatrix(AutoDifferentiableValue):
    '''Auto Differentiable Atomic Matrix'''

    def __init__(self, value: np.ndarray | None):
        super().__init__()
        self.value = value
    
    def clear_gradients(self):
        self.gradient = None
    
    def calculate_value(self) -> None | np.ndarray | float:
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)


class ADVMatMul(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Multiplication Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs @ rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.lhs.value == None or self.rhs.value == None:
            self.lhs.clear_gradients()
            self.rhs.clear_gradients()
            return

        self.lhs.calculate_backward_gradients(_upstream @ self.rhs.value.T)
        self.rhs.calculate_backward_gradients(self.lhs.value.T @ _upstream)


class ADVMatAdd(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Addition Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs + rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream)
        self.rhs.calculate_backward_gradients(_upstream)


class ADVMatSub(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Subtraction Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs - rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.lhs.value == None or self.rhs.value == None:
            self.lhs.clear_gradients()
            self.rhs.clear_gradients()
            return

        self.lhs.calculate_backward_gradients(_upstream)
        self.rhs.calculate_backward_gradients(-_upstream)


class ADVMatElMul(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Multiplication Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs * rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.lhs.value == None or self.rhs.value == None:
            self.lhs.clear_gradients()
            self.rhs.clear_gradients()
            return

        self.lhs.calculate_backward_gradients(_upstream * self.rhs.value)
        self.rhs.calculate_backward_gradients(_upstream * self.lhs.value)


class ADVMatElDiv(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Division Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs / rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.lhs.value == None or self.rhs.value == None:
            self.lhs.clear_gradients()
            self.rhs.clear_gradients()
            return

        self.lhs.calculate_backward_gradients(_upstream / self.rhs.value)
        self.rhs.calculate_backward_gradients(-_upstream * self.lhs.value / (self.rhs.value ** 2))


class ADVMatElPow(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Power Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> None | np.ndarray | float:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()

        if lhs == None or rhs == None:
            self.value = None
            return None

        self.value = lhs ** rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.lhs.value == None or self.rhs.value == None or self.value == None:
            self.lhs.clear_gradients()
            self.rhs.clear_gradients()
            return

        self.lhs.calculate_backward_gradients(_upstream * self.rhs.value * self.value / self.lhs.value)
        self.rhs.calculate_backward_gradients(_upstream * self.value * np.log(self.lhs.value))


class ADVMatTrans(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Transpose Node'''

    def __init__(self, opr: AutoDifferentiableValue):
        super().__init__()
        self.opr = opr
    
    def clear_gradients(self):
        self.gradient = None
        self.opr.clear_gradients()

    def calculate_value(self) -> None | np.ndarray | float:
        opr = self.opr.calculate_value()

        if opr == None:
            self.value = None
            return None

        self.value = opr.T
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.opr.value == None:
            self.opr.clear_gradients()
            return

        self.opr.calculate_backward_gradients(_upstream.T)


class ADVBroadcastTo(AutoDifferentiableValue):
    '''Explicit node to broadcast an array to a target shape'''

    def __init__(self, source: AutoDifferentiableValue, target_shape: tuple | None):
        super().__init__()
        self.source = source
        self.original_shape = None
        self.target_shape = target_shape
    
    def clear_gradients(self):
        self.gradient = None
        self.source.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        source = self.source.calculate_value()

        if source == None or self.target_shape == None:
            self.value = None
            return None

        self.original_shape = source.shape
        self.value = np.broadcast_to(source, self.target_shape)
        
        return self.value

    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.original_shape == None:
            self.source.clear_gradients()
            return

        local_grad = _upstream.copy()
        
        extra_dims = len(local_grad.shape) - len(self.original_shape)
        if extra_dims > 0:
            local_grad = np.sum(local_grad, axis=tuple(range(extra_dims)))
            
        for axis, dim in enumerate(self.original_shape):
            if dim == 1 and local_grad.shape[axis] > 1:
                local_grad = np.sum(local_grad, axis=axis, keepdims=True)
                
        self.source.calculate_backward_gradients(local_grad)


class ADVReLU(AutoDifferentiableValue):
    '''Auto Differentiable ReLU Node'''

    def __init__(self, inputs: AutoDifferentiableValue):
        super().__init__()
        self.inputs = inputs

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        inputs = self.inputs.calculate_value()

        if inputs == None:
            self.value = None
            return None

        self.value = np.maximum(inputs, 0)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.inputs.value == None:
            self.inputs.clear_gradients()
            return

        self.inputs.calculate_backward_gradients(_upstream * (self.inputs.value > 0).astype(float))


class ADVSigmoid(AutoDifferentiableValue):
    '''Auto Differentiable Sigmoid Node'''

    def __init__(self, inputs: AutoDifferentiableValue):
        super().__init__()
        self.inputs = inputs

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        inputs = self.inputs.calculate_value()

        if inputs == None:
            self.value = None
            return None

        self.value = 1 / (1 + np.exp(-inputs))
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.value == None:
            self.inputs.clear_gradients()
            return

        sigmoid = self.value
        reciprocal = 1 - sigmoid

        self.inputs.calculate_backward_gradients(_upstream * sigmoid * reciprocal)


class ADVTanh(AutoDifferentiableValue):
    '''Auto Differentiable Tanh Node'''

    def __init__(self, inputs: AutoDifferentiableValue):
        super().__init__()
        self.inputs = inputs
        self._pos = None
        self._neg = None

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        inputs = self.inputs.calculate_value()

        if inputs == None:
            self.value = None
            return None

        self._pos = np.exp(inputs)
        self._neg = np.exp(-inputs)
        self.value = (self._pos - self._neg) / (self._pos + self._neg)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.value == None:
            self.inputs.clear_gradients()
            return

        self.inputs.calculate_backward_gradients(_upstream * (1 - self.value ** 2))


class ADVSoftmax(AutoDifferentiableValue):
    '''Auto Differentiable Softmax Node'''

    def __init__(self, logits: AutoDifferentiableValue):
        super().__init__()
        self.logits = logits
    
    def clear_gradients(self):
        self.gradient = None
        self.logits.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        logits = self.logits.calculate_value()

        if logits == None:
            self.value = None
            return None

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)

        self.value = probabilities
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.value == None:
            self.logits.clear_gradients()
            return
        
        sum_term = np.sum(self.value * _upstream, axis=1, keepdims=True)
        local_grad = self.value * (_upstream - sum_term)
        
        self.logits.calculate_backward_gradients(local_grad)


class ADVMeanSquaredError(AutoDifferentiableValue):
    '''Auto Differentiable MSE Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray | None):
        super().__init__()
        self.predictions = predictions
        self.targets = targets
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        predictions = self.predictions.calculate_value()

        if self.targets == None or self.predictions.value == None:
            self.value = None
            return None

        self.value = np.mean((self.targets - predictions) ** 2)

        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.targets == None or self.predictions.value == None:
            self.predictions.clear_gradients()
            return
        
        N = self.predictions.value.size
        local_grad = 2 * (self.predictions.value - self.targets) / N
        
        self.predictions.calculate_backward_gradients(local_grad * _upstream)


class ADVBinaryCrossEntropy(AutoDifferentiableValue):
    '''Auto Differentiable Binary Cross Entropy Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray | None):
        super().__init__()
        self.predictions = predictions
        self.targets = targets
        self.epsilon = 1e-7
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        predictions = self.predictions.calculate_value()

        if self.targets == None or self.predictions.value == None:
            self.value = None
            return None

        safe_preds = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        pos_error = self.targets * np.log(safe_preds)
        neg_error = (1 - self.targets) * np.log(1 - safe_preds)

        self.value = -np.mean(pos_error + neg_error)

        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.targets == None or self.predictions.value == None:
            self.predictions.clear_gradients()
            return

        safe_preds = np.clip(self.predictions.value, self.epsilon, 1.0 - self.epsilon)
        numerator = safe_preds - self.targets
        denominator = safe_preds * (1 - safe_preds)
        
        N = self.predictions.value.size
        local_grad = (numerator / denominator) / N

        self.predictions.calculate_backward_gradients(local_grad * _upstream)


class ADVCategoricalCrossEntropy(AutoDifferentiableValue):
    '''Auto Differentiable Categorical Cross Entropy Loss Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray | None):
        super().__init__()
        self.predictions = predictions
        self.targets = targets
        self.epsilon = 1e-7
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> None | np.ndarray | float:
        predictions = self.predictions.calculate_value()

        if self.targets == None or self.predictions.value == None:
            self.value = None
            return None

        safe_preds = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        batch_losses = -np.sum(self.targets * np.log(safe_preds), axis=1)

        self.value = np.mean(batch_losses)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):          
        _upstream = super().calculate_backward_gradients(_upstream)

        if self.targets == None or self.predictions.value == None:
            self.predictions.clear_gradients()
            return
        
        batch_size = self.targets.shape[0]
        safe_preds = np.clip(self.predictions.value, self.epsilon, 1.0 - self.epsilon)
        
        local_grad = -(self.targets / safe_preds) / batch_size
        self.predictions.calculate_backward_gradients(local_grad * _upstream)