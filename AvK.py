import numpy as np
import os

def Adam_parameters_init(node, require_grad:bool, dtype:np.ndarray)->None:
    if require_grad:
        node._m = np.zeros_like(node.grad, dtype)
        node._v = node._m.copy()
    return None

def reverse_broardcast(node, grad: np.ndarray):
    while grad.ndim > node.ndim:
        grad = grad.sum(axis=0)         
    for i,k in zip(range(len(node.shape)), node.shape):
        if k == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def update_gradient(node, grad: np.ndarray) -> None:
    assert isinstance(node, Tensor), "only AvK.Tensor type for node"
    try:
        node.grad += grad #normal
        return
    except: 
        grad = reverse_broardcast(node, grad)
        node.grad += grad

def normal_xavier(n_inp, n_outp) -> np.float_:
    return np.sqrt((2) / (n_inp+n_outp))

def reverse_transpose(x: np.ndarray, axes):
    index_changed = {O:T for O,T in zip(range(len(axes)), axes)}
    
    origin_index = {T:O for O,T in index_changed.items()}
    origin_index = dict(sorted(origin_index.items()))
    origin_index = tuple(origin_index.values())

    return x.transpose(origin_index)


def type_checking(data, dtype) -> np.ndarray:
    if isinstance(data, dtype):
        data = data
    else:
        data = np.array(data, dtype=dtype)
    if data.shape != ():
        grad = np.zeros_like(data,dtype=dtype) #0
    else:
        grad = 0   
    return data, grad



def initialize_params(initialization, in_dims, out_dims, req_bias, require_grad) -> np.ndarray:
    if initialization == 'standard normal':
        Weights = Tensor(np.random.randn(in_dims, out_dims) * normal_xavier(in_dims, out_dims),\
                                _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.random.randn(1,out_dims) * normal_xavier(1,out_dims),\
                                _name='Bias', require_grad = require_grad)
            
    elif initialization == 'standard':
        Weights = Tensor(np.random.randn(in_dims, out_dims), _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.random.randn(1,out_dims), _name='Bias', require_grad = require_grad)
            
    elif initialization == 'uniform':
        Weights = Tensor(np.ones((in_dims, out_dims)) * normal_xavier(in_dims, out_dims),\
                                _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.zeros((1,out_dims)) * normal_xavier(in_dims, out_dims),\
                                _name='Bias', require_grad = require_grad)
    else:
        raise Exception('initialization method is invalid')
    
    return Weights, Bias


class Optimization:
    def __init__(self, optimization_method, alpha,
                 lr, beta1, beta2, lamda):
        self.optimization_method = optimization_method
        self._gradient_calculator = lambda: None
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamda = lamda
        
    def GD(self, node) -> np.ndarray:
        node.grad = np.round(node.grad, decimals=10)
        return node.grad * -self.lr
    
    def MGD(self, node) -> np.ndarray:
        node.grad = np.round(node.grad, decimals=10)
        return (node.grad * -self.lr) + (self.alpha * node._m) 

    def Adam(self, node) -> np.ndarray:
        node.grad = np.round(node.grad, decimals=10)
        first_m = (self.beta1 * np.round(node._m, decimals=10)) + (1 - self.beta1) * node.grad
        second_m = (self.beta2 * node._v) + (1 - self.beta2) * (node.grad)**2
        #Bias correction
        m_hat = first_m / (1 - self.beta1)
        v_hat = second_m / (1 - self.beta2)
        step = -self.lr * (m_hat / (np.sqrt(v_hat)  + 1e-8))
        
        return first_m, second_m, step
    
    def AdamW(self, node) -> np.ndarray:
        node.grad = np.round(node.grad, decimals=10)
        first_m = (self.beta1 * np.round(node._m, decimals=10)) + (1 - self.beta1) * node.grad
        second_m = (self.beta2 * node._v) + (1 - self.beta2) * (node.grad)**2
        # second_m = np.maximum(second_m, node._v)
        #Bias correction
        m_hat = first_m / (1 - self.beta1)
        v_hat = second_m / (1 - self.beta2)
        step = - self.lr *(m_hat / ((np.sqrt(v_hat)) + 1e-8)) - self.lr*(self.lamda * node.data)
        
        return first_m, second_m, step

class Optimizer(Optimization):
    def __init__(self, optimization_method = 'GD', alpha = 0,
                 lr = 0.1, beta1 = 0.9, beta2 = 0.999, lamda = 0.01):
        super().__init__(optimization_method, alpha, lr, beta1, beta2, lamda)
        self.backward_pass = lambda: None
        self.num_parameters = 0
        
        if self.optimization_method == 'GD':
            self._gradient_calculator = self.GD
            self.backward_pass = self.GD_updater
        
        elif self.optimization_method == 'MGD':
            self._gradient_calculator = self.MGD
            self.backward_pass = self.MGD_updater
            
        elif self.optimization_method == 'Adam':
            self._gradient_calculator = self.Adam
            self.backward_pass = self.Adam_updater
        
        elif self.optimization_method == 'AdamW':
            self._gradient_calculator = self.AdamW
            self.backward_pass = self.AdamW_updater
        else:
            raise Exception('optimization_method is invalid')
    
    def n_parameters(self):
        print(f'numbers of parameters: {self.num_parameters / 1e+6}m')
    
    def MGD_updater(self, topo) -> None:
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = None
            if node.require_grad:
                step = self._gradient_calculator(node)
                node.data += step
                node._m = step
                
                self.num_parameters += node.data.size
            # elif not node.retain_grad:
            #     pre_node = node
            
    def GD_updater(self, topo) -> None:
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = None
            if node.require_grad:
                step = self._gradient_calculator(node)
                node.data += step 
                
                self.num_parameters += node.data.size
            # elif not node.retain_grad:
            #     pre_node = node
                
    def Adam_updater(self, topo) -> None:
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = None
            if node.require_grad:
                node._m, node._v, step = self._gradient_calculator(node)
                node.data += step
                
                self.num_parameters += node.data.size
            # elif not node.retain_grad:
            #     pre_node = node
                            
    def AdamW_updater(self, topo) -> None:
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = None
            if node.require_grad:
                node._m, node._v, step = self._gradient_calculator(node)
                node.data += step 
                
                self.num_parameters += node.data.size
            # elif not node.retain_grad:
            #     pre_node = node
                         
                         
                         
    def zeros_grad(self, topo) -> None:
        for node in reversed(topo):
            if node.shape != ():
                node.grad = np.zeros_like(node.data,dtype=node.dtype) #0
            else:
                node.grad = 0.0
    
    def __repr__(self):
        return f'optimization_method: {self.optimization_method}\
            \nlr: {self.lr}\talpha: {self.alpha}\tbeta1: {self.beta1}\tbeta2: {self.beta2}\tlamda: {self.lamda}'
            
            
def concat(Tensor_list: list):
    act_data = np.array([Tensor.data for Tensor in Tensor_list])
    output = Tensor(act_data, (tuple(Tensor_list)), _op = f'cat')
    def _backward():
        for tensor, i in zip(Tensor_list, range(len(Tensor_list))):
            tensor.grad += np.ones_like(tensor.data) * output.grad[i] #(nh, B, T, n_dim)
    output._backward = _backward
    return output


    
class Liner:
    def __init__(self, in_dims: int, out_dims: int, req_bias: bool = False, nonliner = False,\
        act_func = 'relu', require_grad = True, initialization = 'standard normal'):
        self.req_bias = req_bias
        
        self.require_grad = require_grad
        self.nonliner = nonliner
        self.Weights, self.Bias = initialize_params(initialization, in_dims, out_dims, req_bias, require_grad)

        self.act_func = act_func
        
    def __call__(self, x):
        if self.req_bias:
            act = (x @ self.Weights) + self.Bias
        else:
            act = (x @ self.Weights)
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act
        
class DropOut:
    def __init__(self, p = 1) -> None:
        self.p = p
        self.scale = 1/(1-self.p) if self.p < 1 else 1
    def __call__(self, X, no_drop_out = False):
        if not no_drop_out:
            die_out_value = np.random.binomial(1, self.p, size = X.shape[1:])
            return X * self.scale * die_out_value 
        return X 
    
    def __repr__(self):
        return f'act func: {self.act_func}\nnon liner: {self.nonliner}\
            \nrequire grad: {self.require_grad}\
            \nWeights: {self.Weights.shape}\nBias: {self.Bias.shape}'

class Tensor:
    def __init__(self, data, _children = (), _op='', _name='None', dtype=np.float32, require_grad = False, retain_grad=False):
        self.data, self.grad = type_checking(data, dtype)

        self.dtype = dtype
        self._backward = lambda: None
        self._prev = _children
        
        self._op = _op
        self._name = _name
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        self.require_grad = require_grad
        self.retain_grad = retain_grad
        
        Adam_parameters_init(self, self.require_grad, self.dtype)

    def data_export(self, path='./'):
        DATA_SET = []
        #build forward topo
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        #reverse for backward topo, if req_grad then add data to DATA_SET
        for node in reversed(topo):
            if node.require_grad:
                DATA_SET.append(node.data)
        
        # directory = os.path.split(path)[0]
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #     print(f"The new directory named {directory} is created!\n")
        np.savez(path, *DATA_SET)
        
        print(f'DATA SET EXPORT SUCCESS! -> path: {path}\n')
        
        
    def load_data(self, path='./'):
        print(path)
        try:
            DATA_SET = np.load(path, allow_pickle=True) 
        except:
            print('LOAD FAILED! -> DATA SET directory path is not exist!\n')
            return 
        print(f'DATA SET LOAD SUCCESS! -> DATA SET is passing through computation tree!\npath: {path}\n')
        #build forward topo
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        #reverse for backward topo, if req_grad then set node data to correct set
        i = 0
        for node in reversed(topo):
            if node.require_grad:
                node.data = DATA_SET[f'arr_{i}']
                i+=1
                
                
    def __repr__(self):
        return f'Shape: {self.shape}\tdtype: {self.data.dtype}\trequire grad: {self.require_grad}\t_op: {self._op}\t_name: {self._name}\
        \nData:\n {self.data}'
    
    def forward_graph(self, file_name:str):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        try: 
            os.remove(f"{file_name}.txt")
        except:
            pass
        f = open(f"{file_name}.txt", "a")
        for node in topo:
            _name = node._name if node._name != 'None' else ''
            _op = node._op if node._op != '' else 'INP'
            
            if node.shape != ():
                f.write(f'[{_op}] {node.shape} {_name} ->  ')
            else:
                f.write(f'[{_op}]scalar ({node.data}) {_name} ->  ')
        f.close()
           
    def backward(self, optimizer: Optimizer = None, update=False, zeros_grad=True):
        assert isinstance(optimizer, Optimizer), 'optimizer is not an Optimizer type class'
        optimizer.num_parameters = 0
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        if zeros_grad:
            optimizer.zeros_grad(topo)

        if update:
            self.grad = np.ones_like(self.data)
            optimizer.backward_pass(topo)

        else:
            self.grad = np.ones_like(self.data)
            for node in reversed(topo):
                node._backward()

    def copy(self):
        return Tensor(self.data, ())

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other - self

    def __sub__(self, other): 
        return self + (-other)
    
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            update_gradient(self, output.grad)
            update_gradient(other, output.grad)
            
        output._backward = _backward
        return output
    
    def __abs__(self):
        x = self.data
        output = Tensor(abs(x))
        def _backward():
            update_gradient(self, (x/abs(x)) * output.grad)
        output._backward = _backward
        return output
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            update_gradient(self, (other * (self.data ** (other - 1)) * output.grad))
        output._backward = _backward
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, (self, other), '*')   
        def _backward():
            update_gradient(self, (other.data * output.grad))
            update_gradient(other, (self.data * output.grad))
                
        output._backward = _backward
        return output
    
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if len(self.shape) == len(other.shape) == 2:#(3,3)@(3,3)=(3,3)
                self.grad += output.grad @ other.data.T
                other.grad += self.data.T @ output.grad
                
            elif len(self.shape) == 3 and len(other.shape) == 3:#(3,3,3)@(3,3,3) = (3,3,3)
                self.grad += output.grad @ other.data.transpose(0,2,1)
                other.grad += (self.data.transpose(0,2,1) @ output.grad)
            
            elif len(self.shape) == 3 and len(other.shape) == 2:
                self.grad += output.grad @ other.data.T
                other.grad += (self.data.transpose(0,2,1) @ output.grad).sum(axis=0)
                
            else:
                assert f'4d or more array is not supported!'

        
        output._backward = _backward
        return output
    
    def __getitem__(self, idx):
        output = Tensor(self.data[idx], (self, ), 'index')
        def _backward():
            capsule = np.zeros_like(self.data)
            capsule[idx] = output.grad
            update_gradient(self, capsule)
        output._backward = _backward
        return output

    
    def max(self, axis = -1, keepdims = False):
        act_data = np.max(self.data, axis=axis, keepdims=keepdims)
        output = Tensor(act_data, (self,), f'max')
        def _backward():
            update_gradient(self, np.where(self.data == act_data, 1, 0) * output.grad)
        output._backward = _backward
        return output
    
    
    def sum(self, axis = -1, keepdims = False):
        act_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        output = Tensor(act_data, (self,), f'sum')
        def _backward():
            update_gradient(self, (np.ones_like(act_data) * output.grad))
        output._backward = _backward
        return output
    
    
    def exp(self, eps = 0):
        output = Tensor(np.exp(self.data + eps), (self,), f'exp')
        def _backward():
            update_gradient(self, output.data * output.grad)
        output._backward = _backward
        return output
    
    
    def log(self):
        output = Tensor(np.log(self.data), (self,), f'log')
        def _backward():
            update_gradient(self, (self.data)**-1 * output.grad)
        output._backward = _backward
        return output
    
    def mean(self, axis=-1, keepdims=True):
        sum = self.sum(axis, keepdims)
        mean = sum / len(self.data[..., 0][0])
        return mean
    
    def var(self, axis=-1, ddof=0, keepdims=True):
        mean = self.mean(axis)
        MSE = ((self - mean)**2).sum(axis, keepdims)
        N = self.shape[-1] - ddof
        std_output = (MSE/ N)
        return std_output
    
    def std(self, axis=-1, ddof=1, keepdims=True):
        std = self.var(axis, ddof, keepdims) 
        var_output = (std+1e-5)**0.5
        return var_output
    
    def LayerNorm(self, ddof = 0):
        x = self
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, ddof = ddof, keepdims=True)
        NORM = (x - mean) / (std + 1e-5)
        return NORM
    
    def where(self, condition, if_True, if_False):
        output = Tensor(np.where(condition, if_True, if_False), (self,), 'where')
        
        def _backward():
            update_gradient(self, np.where(condition, 0, 1) * output.grad)
            
        output._backward = _backward
        return output
    
    def tril(self, k = 0):
        output = Tensor(np.tril(self.data, k), (self,), 'tril')
        
        def _backward():
           update_gradient(self, np.where(output.data == 0, 0, 1) * output.grad)
        
        output._backward = _backward
        return output
    
    def transpose(self, *axes):
        output = Tensor(self.data.transpose(axes), (self,), f'T')
        def _backward():
            update_gradient(self, reverse_transpose(output.grad, axes))
        output._backward = _backward
        return output
    
    def reshape(self, *new_shape):
        output = Tensor(self.data.reshape(new_shape), (self,), f'reshape')
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output
    
    def squeeze(self):
        output = Tensor(np.squeeze(self.data), (self,), f'squeeze')
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output
    
    def flatten(self):
        output = Tensor(self.data.flatten(), (self,), f'flatten')
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output


    def cross_entropy_loss(self, target):
        x = self.data.copy()
        if len(x.shape) == 4:
            raise Exception(f'4d array, not supported!')
        elif len(x.shape) == 3:
            B = x.shape[0]
            T = x.shape[1]
        elif len(x.shape) == 2:
            B = x.shape[0]
            T = 1
        else:
            raise Exception('cross entropy error', f'data shape {self.shape}')
        target =  np.array(target)
        probs = softmax(x)
        logprobs = np.vstack(log_softmax(self))
        loss = -logprobs[range(B*T), target.flatten()].mean()

        output = Tensor(loss, _children=(self,), _op = f'cross entropy loss')

        def _backward():
            dlogits = np.vstack(probs)
            dlogits[range(B*T), target.flatten()] -= 1
            dlogits = dlogits.reshape(probs.shape)
            
            self.grad = dlogits

        output._backward = _backward
        return output
    
    def MSE(self, target):
        assert self.data.size == target.data.size, 'x and target have different size( MSE require x and target to have the same size)'

        return ((self.flatten() - target.flatten())**2).sum(axis=(0)) / (self.data.size)
    
    def KL_divergence(self, target):
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        assert self.data.size == target.data.size, 'x and target have different size( KL_divergence require x and target to have the same size)'
        
        probs = self.softmax()
        
        KL_D = ((target / probs).log() * target)
        N = (np.sum([i for i in probs.shape]))
        AXIS = tuple([i for i in range(probs.data.ndim)])

        return KL_D.sum(axis=AXIS) / N
        
    
    def relu(self):
        output = Tensor(np.where(self.data >=0, self.data, 0), (self,), 'ReLU')
        def _backward():
            update_gradient(self, np.where(output.data >0, 1, 0) * output.grad)
        output._backward = _backward

        return output
    
    def leaky_relu(self, alpha = 0.01):
        output = Tensor(np.where(self.data < 0, self.data * alpha, self.data), (self,), 'Leaky_ReLU')
        def _backward():
            update_gradient(self, np.where(output.data > 0, 1, alpha) * output.grad)
        output._backward = _backward
        return output
    
    def soft_plus(self):
        output = Tensor(np.log(1+np.exp(self.data)), (self,), 'Soft plus')
        def _backward():
            update_gradient(self, (1 / (1 + np.exp(-self.data))) * output.grad)
        output._backward = _backward
        return output
    
    def sigmoid(self):
        output = Tensor(1 / (1+np.exp(-self.data)), (self,), 'Sigmoid')
        def _backward():
            update_gradient(self, (output.data * (1-output.data)) * output.grad)
        output._backward = _backward
        return output
    
    def softmax(self, axis=-1):
        self -= self.max(axis, keepdims=True)
        exp = self.exp()
        probs = exp / (exp.sum(axis, keepdims=True)) 
        probs._op = 'softmax'
        return probs
    
    
    def log_softmax(self, axis=-1): 
        self -= self.max(axis, keepdims=True)
        
        if len(self.shape) == 2:
            out = self - self.exp().sum(axis).log().reshape(self.shape[0],  -1)
        elif len(self.shape) == 3:
            out = self - self.exp().sum(axis).log().reshape(self.shape[1],  -1)
        else:
            raise Exception(f'Invalid log_softmax operation on {self}')
        
        return out
    
def softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    probs = exp / (np.sum(exp, axis, keepdims=True)) 
    return probs
        
def log_softmax(input, axis=-1): 
    data = input.data.copy()
    data -= np.max(data, axis, keepdims=True)
    
    if len(data.shape) == 2:
        out = data - np.log(np.sum(np.exp(data), axis)).reshape(data.shape[0],  -1)
    elif len(data.shape) == 3:
        out = data - np.log(np.sum(np.exp(data), axis)).reshape(data.shape[0], data.shape[1], -1)
    else:
        raise Exception(f'Invalid log_softmax operation on {input}')
    
    return out

def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

pass


class Conv():
    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride=1, padding=0):

        self.d_X, self.h_X, self.w_X = X_dim

        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter
        self.stride, self.padding = stride, padding

        self.W = Tensor(np.random.randn(
            n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.), require_grad=True, _name = 'Kernal')##
        self.b = Tensor(np.zeros((self.n_filter, 1)), require_grad=True, _name = 'Bias')##
        self.params = [self.W, self.b]

        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1
        
        print(self.h_out)
        print(f'{self.h_X=}, {h_filter=}, {padding=}, {stride=}')

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.n_filter, self.h_out, self.w_out)

    def __call__(self, X):
        assert isinstance(X, Tensor), "only AvK.Tensor type for X(input)"
        X_data = X.data
        self.n_X = X.shape[0]

        self.X_col =  (im2col_indices(
            X_data, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding))
        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row.data @ self.X_col + self.b.data##
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        
        out = Tensor(out, (self.W, self.b, X), 'CONV')###
        
        def _backward():
            dX, dW, db = self.backward(out.grad)
            update_gradient(X, dX)
            update_gradient(self.W, dW)
            update_gradient(self.b, db)
        out._backward = _backward
        return out

    def backward(self, dout):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.data.reshape(self.n_filter, -1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.h_filter,
                            self.w_filter, self.padding, self.stride)

        return dX, dW, db


class Maxpool():
    def __init__(self, X_dim, size, stride=2):

        self.d_X, self.h_X, self.w_X = X_dim

        self.params = []

        self.size = size
        self.stride = stride

        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.d_X, self.h_out, self.w_out)

    def __call__(self, X):
        X_data = X.data
        self.n_X = X.shape[0]
        X_reshaped = X_data.reshape(
            X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        self.X_col = im2col_indices(
            X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indexes = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]

        out = out.reshape(self.h_out, self.w_out, self.n_X,
                          self.d_X).transpose(2, 3, 0, 1)
        
        out = Tensor(out, (X,), 'MAX POOL')###
        def _backward():
            dX, _  = self.backward(out.grad)
            update_gradient(X, dX)
        out._backward = _backward
        return out

    def backward(self, dout):

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (self.n_X * self.d_X, 1, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.size,
                            self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.n_X, self.d_X, self.h_X, self.w_X)
        return dX, []



class Convolution2D:
    def __init__(self, input_shape: tuple, n_kernels: int, kernel_size: int, kernels_stride = 1,  padding = 0,\
        pooling = False, pooling_kernals = 2,  pool_stride = 2,\
        nonliner=False, act_func='relu'):
        
        self.CONV = Conv(input_shape, n_kernels, kernel_size, kernel_size, stride=kernels_stride, padding= padding)
        self.Pool = Maxpool(self.CONV.out_shape, pooling_kernals, pool_stride) if pooling else None
        
        self.nonliner = nonliner
        self.pooling = pooling
        
        self.act_func = act_func
        self.out_shape = self.Pool.out_shape if pooling else self.CONV.out_shape
        
    def __call__(self, x):
        act = self.Pool(self.CONV(x)) if self.pooling else self.CONV(x)
        
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act


class Upsampling:
    def __init__(self, X_dim, scale_factor = 2) -> None:
        self.B, self.T, self.C = X_dim
        self.B_out, self.T_out, self.C_out = self.B, self.T * scale_factor, self.C * scale_factor
        self.scale_factor = scale_factor
        
        self.out_shape = (self.B_out, self.T_out, self.C_out)
        
    def __call__(self, X):
        out = Tensor(np.repeat(X.data, self.scale_factor, axis=-2).repeat(self.scale_factor, axis=-1), (X,), 'up sampling')
        
        def _backward():
            dX = self.backward(out.grad)
            update_gradient(X, dX)
        out._backward = _backward
        return out
    
    def backward(self, glob_grad):
        first_m = np.add.reduceat(glob_grad, np.mgrid[0:self.T_out: self.scale_factor], axis=-2)
        second_m = np.add.reduceat(first_m, np.mgrid[0:self.C_out: self.scale_factor], axis=-1)
        
        return second_m
            