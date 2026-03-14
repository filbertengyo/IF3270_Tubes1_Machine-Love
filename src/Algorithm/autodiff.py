from abc import ABC, abstractmethod
import numpy as np

class AutoDifferentiableValue(ABC):
    '''Abstract base class for all values that can be auto differentiable as part of a computation graph'''

    def __init__(self, value: np.ndarray):
        self.gradient: None | np.ndarray = None
        self.value: np.ndarray = value.copy()
    
    @abstractmethod
    def clear_gradients(self):
        raise NotImplementedError()
    
    @abstractmethod
    def calculate_value(self) -> np.ndarray:
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

    def __init__(self, value: np.ndarray):
        super().__init__(value)
    
    def clear_gradients(self):
        self.gradient = None
    
    def calculate_value(self) -> np.ndarray:
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)


class ADVMatMul(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Multiplication Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value @ rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs @ rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream @ self.rhs.value.T)
        self.rhs.calculate_backward_gradients(self.lhs.value.T @ _upstream)


class ADVMatAdd(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Addition Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value + rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs + rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream)
        self.rhs.calculate_backward_gradients(_upstream)


class ADVMatSub(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Subtraction Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value - rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs - rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream)
        self.rhs.calculate_backward_gradients(-_upstream)


class ADVMatElMul(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Multiplication Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value * rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs * rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream * self.rhs.value)
        self.rhs.calculate_backward_gradients(_upstream * self.lhs.value)


class ADVMatElDiv(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Division Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value / rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs / rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream / self.rhs.value)
        self.rhs.calculate_backward_gradients(-_upstream * self.lhs.value / (self.rhs.value ** 2))


class ADVMatElPow(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Element-wise Power Node'''

    def __init__(self, lhs: AutoDifferentiableValue, rhs: AutoDifferentiableValue):
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(lhs.value ** rhs.value)
    
    def clear_gradients(self):
        self.gradient = None
        self.lhs.clear_gradients()
        self.rhs.clear_gradients()

    def calculate_value(self) -> np.ndarray:
        lhs = self.lhs.calculate_value()
        rhs = self.rhs.calculate_value()
        self.value = lhs ** rhs
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.lhs.calculate_backward_gradients(_upstream * self.rhs.value * self.value / self.lhs.value)
        self.rhs.calculate_backward_gradients(_upstream * self.value * np.log(self.lhs.value))


class ADVMatTrans(AutoDifferentiableValue):
    '''Auto Differentiable Matrix Transpose Node'''

    def __init__(self, opr: AutoDifferentiableValue):
        self.opr = opr
        super().__init__(opr.value.T)
    
    def clear_gradients(self):
        self.gradient = None
        self.opr.clear_gradients()

    def calculate_value(self) -> np.ndarray:
        opr = self.opr.calculate_value()
        self.value = opr.T
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.opr.calculate_backward_gradients(_upstream.T)


class ADVBroadcastTo(AutoDifferentiableValue):
    '''Explicit node to broadcast an array to a target shape'''

    def __init__(self, source: AutoDifferentiableValue, target_shape: tuple):
        self.source = source
        self.original_shape = source.value.shape
        self.target_shape = target_shape
        super().__init__(np.broadcast_to(source.value, target_shape))
    
    def clear_gradients(self):
        self.gradient = None
        self.source.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        source = self.source.calculate_value()
        self.original_shape = source.shape
        self.value = np.broadcast_to(source, self.target_shape)
        
        return self.value

    def calculate_backward_gradients(self, _upstream = None):
            _upstream = super().calculate_backward_gradients(_upstream)

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
        self.inputs = inputs
        super().__init__(np.maximum(inputs.value, 0))

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        inputs = self.inputs.calculate_value()
        self.value = np.maximum(inputs, 0)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.inputs.calculate_backward_gradients(_upstream * (self.inputs.value > 0).astype(float))


class ADVSigmoid(AutoDifferentiableValue):
    '''Auto Differentiable Sigmoid Node'''

    def __init__(self, inputs: AutoDifferentiableValue):
        self.inputs = inputs
        super().__init__(1 / (1 + np.exp(-inputs.value)))

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        inputs = self.inputs.calculate_value()
        self.value = 1 / (1 + np.exp(-inputs))
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        sigmoid = self.value
        reciprocal = 1 - sigmoid
        self.inputs.calculate_backward_gradients(_upstream * sigmoid * reciprocal)


class ADVTanh(AutoDifferentiableValue):
    '''Auto Differentiable Tanh Node'''

    def __init__(self, inputs: AutoDifferentiableValue):
        self.inputs = inputs
        self._pos = np.exp(inputs.value)
        self._neg = np.exp(-inputs.value)
        super().__init__((self._pos - self._neg) / (self._pos + self._neg))

    def clear_gradients(self):
        self.gradient = None
        self.inputs.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        inputs = self.inputs.calculate_value()
        self._pos = np.exp(inputs)
        self._neg = np.exp(-inputs)
        self.value = (self._pos - self._neg) / (self._pos + self._neg)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        self.inputs.calculate_backward_gradients(_upstream * (1 - self.value ** 2))


class ADVSoftmax(AutoDifferentiableValue):
    '''Auto Differentiable Softmax Node'''

    def __init__(self, logits: AutoDifferentiableValue):
        self.logits = logits
        
        shifted_logits = logits.value - np.max(logits.value, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        
        super().__init__(probabilities)
    
    def clear_gradients(self):
        self.gradient = None
        self.logits.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        logits = self.logits.calculate_value()

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)

        self.value = probabilities
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        
        sum_term = np.sum(self.value * _upstream, axis=1, keepdims=True)
        local_grad = self.value * (_upstream - sum_term)
        
        self.logits.calculate_backward_gradients(local_grad)


class ADVMeanSquaredError(AutoDifferentiableValue):
    '''Auto Differentiable MSE Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray):
        self.predictions = predictions
        self.targets = targets
        super().__init__(np.mean((targets - predictions.value) ** 2))
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        predictions = self.predictions.calculate_value()
        self.value = np.mean((self.targets - predictions) ** 2)
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)
        
        N = self.predictions.value.size
        local_grad = 2 * (self.predictions.value - self.targets) / N
        
        self.predictions.calculate_backward_gradients(local_grad * _upstream)


class ADVBinaryCrossEntropy(AutoDifferentiableValue):
    '''Auto Differentiable Binary Cross Entropy Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray):
        self.predictions = predictions
        self.targets = targets
        self.epsilon = 1e-7
        
        safe_preds = np.clip(predictions.value, self.epsilon, 1.0 - self.epsilon)
        pos_error = targets * np.log(safe_preds)
        neg_error = (1 - targets) * np.log(1 - safe_preds)

        super().__init__(-np.mean(pos_error + neg_error))
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        predictions = self.predictions.calculate_value()

        safe_preds = np.clip(predictions.value, self.epsilon, 1.0 - self.epsilon)
        pos_error = self.targets * np.log(safe_preds)
        neg_error = (1 - self.targets) * np.log(1 - safe_preds)

        self.value = -np.mean(pos_error + neg_error)

        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):
        _upstream = super().calculate_backward_gradients(_upstream)

        safe_preds = np.clip(self.predictions.value, self.epsilon, 1.0 - self.epsilon)
        numerator = safe_preds - self.targets
        denominator = safe_preds * (1 - safe_preds)
        
        N = self.predictions.value.size
        local_grad = (numerator / denominator) / N

        self.predictions.calculate_backward_gradients(local_grad * _upstream)


class ADVCategoricalCrossEntropy(AutoDifferentiableValue):
    '''Auto Differentiable Categorical Cross Entropy Loss Node'''

    def __init__(self, predictions: AutoDifferentiableValue, targets: np.ndarray):
        self.predictions = predictions
        self.targets = targets
        self.epsilon = 1e-7
        
        safe_preds = np.clip(predictions.value, self.epsilon, 1.0 - self.epsilon)
        batch_losses = -np.sum(targets * np.log(safe_preds), axis=1)
        super().__init__(np.mean(batch_losses))
    
    def clear_gradients(self):
        self.gradient = None
        self.predictions.clear_gradients()
    
    def calculate_value(self) -> np.ndarray:
        predictions = self.predictions.calculate_value()

        safe_preds = np.clip(predictions.value, self.epsilon, 1.0 - self.epsilon)
        batch_losses = -np.sum(self.targets * np.log(safe_preds), axis=1)

        self.value = np.mean(batch_losses)
        
        return self.value
    
    def calculate_backward_gradients(self, _upstream = None):          
        _upstream = super().calculate_backward_gradients(_upstream)
        
        batch_size = self.targets.shape[0]
        safe_preds = np.clip(self.predictions.value, self.epsilon, 1.0 - self.epsilon)
        
        local_grad = -(self.targets / safe_preds) / batch_size
        self.predictions.calculate_backward_gradients(local_grad * _upstream)