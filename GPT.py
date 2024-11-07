import numpy as np
from AvK import Tensor, Liner, DropOut, Optimizer, normal_xavier, concat, softmax
import os

def mini_batches(data, max_sequence_len: int = 5, batch_size = 16):
    pos = np.random.randint(low=0, high=len(data) - max_sequence_len, size=(batch_size, ))
    xs = np.array([data[i:i+max_sequence_len] for i in pos])
    ys = np.array([data[i + 1:i+max_sequence_len + 1] for i in pos])
    return xs, ys

def greedy_sampling(model_structure, prompts:list):
    logits = model_structure(prompts, no_drop_out = True)
    probs = np.float64(softmax(logits.data)[0])
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    return probs[-1]


def greedy_generator(model_structure, decode, T:int, prompts:list, max_new_tokens: int, num_samples: int = 1):
    for i in range(max_new_tokens):
        croped_prompts = np.array(prompts)[np.newaxis,:][:, -T:]
        probs = greedy_sampling(model_structure, croped_prompts)
        out = np.argmax(np.random.multinomial(num_samples, probs.ravel()))
        
        prompts.append(out)
    #     os.system('clear')
    #     print(decode(prompts))
    # os.system('clear')
    return prompts


class Beam_particle:
    def __init__(self, idx, prob, value:list, T, cumc_prob=1) -> None:
        self.idx = idx
        self.prob = prob
        
        self.value = value
        self.T = T
        self.cumc_prob = cumc_prob
    def __call__(self, model_structure, beam_width):
        output = []
        next_prob = greedy_sampling(model_structure, np.array(self.value)[np.newaxis, :][:, -self.T:])
        next_idx = np.argsort(next_prob)[::-1][:beam_width]
        
        for i in range(beam_width):
            particle_idx = next_idx[i]
            particle_prob = next_prob[particle_idx]
            particle_value = self.value.copy()
            particle_value.append(particle_idx)
            beam_particle = Beam_particle(particle_idx, particle_prob, particle_value, self.T, cumc_prob= (self.prob*particle_prob) )

            output.append(beam_particle)
        return output
        
def beam_generator_wraper(model_structure, T:int, prompts:list, max_new_tokens: int, beam_width = 3):
    Pos1 = Beam_particle(prompts[-1], 1, prompts, T)(model_structure, beam_width)
    for _ in range(max_new_tokens):
        Beam_pool = [Pos1[i](model_structure, beam_width) for i in range(beam_width)]
        Beam_pool = [particle for particles in Beam_pool for particle in particles]
        Beam_pool.sort(key=lambda x: x.cumc_prob, reverse=True)
        Pos1 = Beam_pool[0:beam_width]

    Pos1.sort(key=lambda x: x.cumc_prob, reverse=True)
    return Pos1[0].value

def getPositionEncoding(shape, n=100):
    P = np.zeros(shape[1::])
    seq_len = shape[1]
    d_model = shape[2]
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
    
class SELF_ATTENTION_HEAD:
    def __init__(self, C, head_dims):
        self.C = C
        self.head_dims = head_dims
        
        self.query = Liner(C, head_dims,initialization='standard normal')
        self.key = Liner(C, head_dims,initialization='standard normal')
        self.value = Liner(C, head_dims,initialization='standard normal')
        
        self.drop_out = DropOut()
    def __call__(self, pos_emb, no_drop_out = False):
        
        Q = self.query(pos_emb)
        K = self.key(pos_emb)
        V = self.value(pos_emb)

        attention = Q @ K.transpose(0,2,1)
        scale = attention * (self.C**-0.5)
        mask_in = scale.where(np.tril(scale.data) == 0, -np.inf, scale.data)

        mask_prob = mask_in.softmax()

        attention_values = self.drop_out(mask_prob, no_drop_out) @ V
        
        return attention_values

        
class MULTIHEAD_ATTENTION:
    def __init__(self, n_heads,C):
        print(f'n heads: {n_heads}, C: {C}')
        assert C % n_heads == 0, f'{C} d_models cannot split into {n_heads} heads!'
        
        self.n_heads = n_heads
        self.head_dims = C//n_heads
        self.C = C

        self.proj = Liner(self.C, self.C, initialization='standard normal')
        self.heads = [SELF_ATTENTION_HEAD(self.C, self.head_dims) for _ in range(self.n_heads)]
        self.drop_out = DropOut()
    
    
    def __call__(self, pos_emb, no_drop_out = False):
        MH = concat([head(pos_emb) for head in self.heads]).transpose(1,2,0,3)
        rMH = self.drop_out(MH.reshape(*pos_emb.shape), no_drop_out)
        
        proj_values = self.proj(rMH)
        return proj_values
    
    
class LAYER_NORM:
    def __init__(self, C):
        self.gamma = Tensor(np.ones((1,C)) * normal_xavier(1, C), require_grad=True,\
                            _name='gamma')
        self.beta = Tensor(np.zeros((1,C)) * normal_xavier(1, C), require_grad=True,\
                          _name = 'beta')
    def __call__(self, x):
        
        NORM = x.LayerNorm()
        
        shift_scale = NORM * self.gamma + self.beta
        
        return shift_scale
        
class EMB_POS_encoding:
    def __init__(self, T,C, vocab):
        self.token_emb_w = Tensor(np.random.randn(vocab, C), require_grad=True)# * normal_xavier(T,C)  #(vocab,C)
    def __call__(self, X):
        token_emb = self.token_emb_w[X]
        emb = token_emb + Tensor(getPositionEncoding(token_emb.shape))
        
        return emb
    
class FeedForward:
    def __init__(self, C):
        self.C = C
        self.leaky_relu = Liner(C, 4*C, initialization='standard normal', nonliner=True, act_func='relu', req_bias=True)
        self.liner = Liner(4*C, C, initialization='standard normal',
                           req_bias=True)
        
        self.drop_out = DropOut()
        
    def __call__(self, x: np.ndarray, no_drop_out = False):
        leaky_relu_values = self.leaky_relu(x)
        liner_values = self.liner(leaky_relu_values)
        return self.drop_out(liner_values, no_drop_out)
    
class BLOCK:
    def __init__(self, B,C, n_heads):
        self.multihead_attention = MULTIHEAD_ATTENTION(n_heads, C)
        # self.layer_norm = LAYER_NORM(C)
        
        self.MH_norm = LAYER_NORM(C)
        self.ffw = FeedForward(C)
        self.ffw_norm = LAYER_NORM(C)
        
    def __call__(self, x, no_drop_out = False):
        x = x + (self.multihead_attention(self.MH_norm(x), no_drop_out))
        x = x + self.ffw(self.ffw_norm(x), no_drop_out)
        return x
    
    
class SEQUENTIAL_BLOCKS:
    def __init__(self, B, C, n_heads, n_layers):
        self.n_layers = n_layers
        
        self.blocks = [BLOCK(B, C, n_heads) for _ in range(n_layers)]
            
    def __call__(self, x, no_drop_out = False):
        for block in self.blocks:
            x = block(x, no_drop_out)
        return x
    
# --------------------------------------
class GPT_MODEL:
    def __init__(self, training_path='', path:str='', batch_sizes = 16, max_sequence_len = 90, \
        emb_dims = 180, n_heads = 4, n_layers = 1, lr = 1e-2, optimization_method = 'MGD') -> None:
        
        self.training_path = training_path
        self.path = str(path)
        
        datas = open(self.training_path, 'r', encoding='utf-8').read()

        self.vocab = sorted(list(set(datas)))
        stoi = {s:i for i,s in enumerate(self.vocab)}
        itos = {i:s for s,i in stoi.items()}


        self.encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        n = int(0.9*len(datas))
        self.train = datas[:n]
        self.valid = datas[n:]
        
        
        self.batch_sizes = int(batch_sizes) #B
        self.max_sequence_len = int(max_sequence_len) #T
        self.emb_dims = int(emb_dims) #C
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.lr = float(lr)

        self.B, self.T, self.C, self.vocab = self.batch_sizes, self.max_sequence_len, self.emb_dims, len(itos)

        self.optimizer = Optimizer(lr=self.lr, optimization_method = optimization_method)

        self.emb_pos_enc = EMB_POS_encoding(self.T, self.C, self.vocab)
        self.sequential_blocks = SEQUENTIAL_BLOCKS(self.B, self.C, self.n_heads, self.n_layers)
        self.layer_norm = LAYER_NORM(self.C)
        self.proj = Liner(self.C, self.vocab, initialization='standard normal',req_bias=True)
        
        print(self.optimizer)
        print(f'dtype: {self.proj.Weights.dtype}\n')
        
        self.recursion = False
        
    def import_vocab(self, vocab):
        self.vocab = vocab
        
    def model_structure(self, X, no_drop_out = False):
        emb = self.emb_pos_enc(X)
        sequential_blocks_values = self.sequential_blocks(emb, no_drop_out)
        norm_value = self.layer_norm(sequential_blocks_values) 
        proj_value = self.proj(norm_value)
        
        return proj_value
    
    def forward_pass(self, epochs:int, datas_export=True, load_datas = True,\
        finetune = False, update = True, zeros_grad = True, backward = True):
        epochs = int(epochs)
        with np.errstate(all = 'raise'):
            for i in range(epochs):
                X,Y = mini_batches(self.encode(self.train), max_sequence_len = self.T, batch_size= self.B)
                # #FORWARD PASS
                proj_value = self.model_structure(X, no_drop_out= False)

                loss = proj_value.cross_entropy_loss(Y)
                
                if finetune and not self.recursion:
                    loss.load_data(self.path)
                    self.recursion = True
                    self.forward_pass(epochs, datas_export, load_datas, finetune, update, zeros_grad, backward=True)
                    return
                elif i == 0 and load_datas and not finetune:
                    loss.load_data(self.path)
                    self.forward_pass(epochs=1, datas_export=False, load_datas=False, finetune=False, update=False, backward=False)
                    return
                
                if backward:
                # #BACKWARD PASS
                    loss.backward(self.optimizer, update, zeros_grad)

                if i %(epochs / 40) == 0:
                    vX,vY = mini_batches(self.encode(self.valid), max_sequence_len = self.T, batch_size= self.B)
                    valid = self.model_structure(vX, no_drop_out=True)
                    valid_loss = valid.cross_entropy_loss(vY)
                    print(f'iter:{i}  train: {loss.data} / valid: {valid_loss.data} / n_parameters: {self.optimizer.num_parameters}')
            
            if datas_export:
                loss.data_export(self.path)
    def inferance(self, max_new_tokens=200):
        inp = input('INP:\n')
        H = [i for i in inp]
        prompts = self.encode(H)
        
        print('\n' + '-----------' +'\n' \
            +self.decode(greedy_generator(self.model_structure, T=self.T, prompts=prompts, max_new_tokens=max_new_tokens, decode=self.decode))\
                + '\n' + '-----------')
        
        