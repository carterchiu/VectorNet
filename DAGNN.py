''' Implements DAG-NNs and vector-weighted DAG-NNs '''

import cupy as cp
import time
from enum import Enum, auto

class ProblemType(Enum):
    CLASS_SINGLE = auto()
    CLASS_ONEHOT = auto()
    REGRESS = auto()

class VecDAGNN:      
    
    def __init__(self, inputs, hidden, outputs, height, expfct, aggfct, problem_type, pop_size, dropout, log_file=None):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.height = height
        self.expfct = expfct
        self.aggfct = aggfct
        self.total = inputs + hidden + outputs
        self.problem_type = problem_type
        self.pop_size = pop_size
        self.dropout = 1 - dropout
        self.log_file = log_file
        self._initialize_nets()
        
    def train(self, x_train, y_train, x_test, y_test, iterations):
        self.minx = -min(x_train.min().item(), x_test.min().item())
        train_in = cp.tile(cp.concatenate((x_train, cp.zeros(shape=(x_train.shape[0], self.hidden + self.outputs))), axis=1), (self.pop_size, 1, 1))
        test_in = cp.tile(cp.concatenate((x_test, cp.zeros(shape=(x_test.shape[0], self.hidden + self.outputs))), axis=1), (self.pop_size, 1, 1))
        for i in range(iterations):
            start_time = time.time()
            y_hat = self._forward_prop(train_in, True)
            fitness, train_loss, train_acc = self.evaluate(y_hat, y_train, self.problem_type)
            self._evolve(fitness)
            test_loss, test_acc = self.evaluate(self._forward_prop(test_in, False), y_test, self.problem_type)[1:]
            print('[GENERATION %i/%i]\n\tTIME: %.2f seconds | TRAIN Loss: %.4f Acc: %.4f | TEST Loss: %.4f Acc: %.4f\n' % (i + 1, iterations, time.time() - start_time, train_loss, train_acc, test_loss, test_acc))
            if self.log_file:
                with open(self.log_file, 'a+') as file:
                    file.write(' '.join([str(s) for s in [i + 1, iterations, time.time() - start_time, train_loss, train_acc, test_loss, test_acc]]) + '\n')
        return test_acc
    
    def _initialize_nets(self):
        self.weights = cp.random.normal(size=(self.pop_size, self.height, self.total, self.total), dtype='single') * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, self.height, 1, 1)) * (cp.random.uniform(size=(self.pop_size, self.height, self.total, self.total), dtype='single') < .5)
        self.weights[:, :, :, self.inputs:] *= cp.sqrt(4 / cp.minimum(cp.arange(self.inputs, self.total), self.inputs + self.hidden))
        self.weights[:, :, :self.inputs, :self.inputs] = 0
        self.weights[:, :, -self.outputs:, -self.outputs:] = cp.tile(cp.diag(cp.diag(cp.ones(shape=(self.outputs, self.outputs), dtype='bool_'))), (self.pop_size, self.height, 1, 1))
        self.biases = cp.random.normal(size=(self.pop_size, self.height, 1, self.hidden + self.outputs), dtype='single') * .5
        
    def _forward_prop(self, inp, training):
        if training:
            mask = self._generate_dropout_mask(inp.shape[1])
        exp_inp = self.expfct(inp, self.height, self.minx)
        fprop = VecDAGNN.hadadot(exp_inp, self.weights)
        for i in range(self.inputs, self.inputs + self.hidden):
            wedge = VecDAGNN.relu(fprop[:, :, :, [i]] + self.biases[:, :, :, [i - self.inputs]]) * (mask[:, [i - self.inputs]] if training else 1)
            fprop[:, :, :, i] = 0
            fprop += VecDAGNN.hadadot(wedge, self.weights[:, :, [i], :]) / self.dropout
        out = fprop[:, :, :, -self.outputs:] + self.biases[:, :, :, -self.outputs:]
        agg_out = self.aggfct(out)
        if self.problem_type == ProblemType.CLASS_SINGLE:
            return VecDAGNN.sigmoid(agg_out)
        elif self.problem_type == ProblemType.CLASS_ONEHOT:
            return VecDAGNN.softmax(agg_out)
        elif self.problem_type == ProblemType.REGRESS:
            return agg_out
    
    def _generate_dropout_mask(self, dim):
        return (cp.random.uniform(size=(dim, self.hidden)) < self.dropout) / self.dropout
    
    def _evolve(self, fitness):
        selected = self._select(fitness)
        self._crossover(selected)
        self._mutate()
        
    def _select(self, fitness):
        order = cp.argsort(fitness)
        return order[(cp.random.uniform(size=(2, self.pop_size)) ** 2 * self.pop_size).astype(int)]
        
    def _crossover(self, selected):
        point = cp.random.randint(1, self.total-1).item()
        self.weights = cp.concatenate((self.weights[selected[0, :], :, :point, :], self.weights[selected[1, :], :, point:, :]), axis=2)
        self.biases = self.biases[selected[1, :], :, :, :] if point <= self.inputs else cp.concatenate((self.biases[selected[0, :], :, :, :point - self.inputs], self.biases[selected[1, :], :, :, point - self.inputs:]), axis=3)

    def _mutate(self):
        weight_mask = cp.random.normal(size=(self.weights.shape), dtype='single') * .05 * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, self.height, 1, 1)) * (cp.random.uniform(size=(self.weights.shape), dtype='single') < .2) * (self.weights != 0)
        new_weights_mask = cp.random.normal(size=(self.weights.shape), dtype='single') * .05 * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, self.height, 1, 1)) * (cp.random.uniform(size=(self.weights.shape), dtype='single') < .05) * (self.weights == 0)
        self.weights += weight_mask + new_weights_mask
        self.weights *= (cp.random.uniform(size=self.weights.shape, dtype='single') < 1)
        self.weights[:, :, :self.inputs, :self.inputs] = 0
        self.weights[:, :, -self.outputs:, -self.outputs:] = cp.tile(cp.diag(cp.diag(cp.ones(shape=(self.outputs, self.outputs), dtype='bool_'))), (self.pop_size, self.height, 1, 1))
        self.biases += cp.random.normal(size=self.biases.shape, dtype='single') * .05 * (cp.random.normal(size=self.biases.shape, dtype='single') < .2)
    
    @staticmethod
    def relu(a):
        return cp.maximum(a, 0)
    
    @staticmethod
    def hadadot(a, b):
        return a @ b
    
    @staticmethod
    def softmax(a):
        return cp.exp(a) / cp.sum(cp.exp(a), axis=2, keepdims=True)
    
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + cp.exp(-a))
    
    @staticmethod
    def evaluate(y_hat, y, problem_type):
        fitness = VecDAGNN.eval_fitness(y_hat, y, problem_type)
        order = cp.argsort(fitness)
        best_loss = fitness[order[0]]
        best_preds = y_hat[order[0]]
        best_acc = VecDAGNN.eval_acc(best_preds, y, problem_type)
        return fitness, best_loss, best_acc
    
    @staticmethod
    def eval_fitness(y_hat, y, problem_type):
        if problem_type in [ProblemType.CLASS_SINGLE, ProblemType.CLASS_ONEHOT]:
            return VecDAGNN.crossentropy(y_hat, y)
        elif problem_type == ProblemType.REGRESS:
            return VecDAGNN.mse(y_hat, y)
    
    @staticmethod
    def crossentropy(y_hat, y):
        return -1 / y_hat.shape[1] * cp.sum(y * cp.log(y_hat + 1e-5) + (1 - y) * cp.log(1 - y_hat + 1e-5), axis=(1,2))
    
    @staticmethod
    def mse(y_hat, y):
        return cp.sum((y_hat - y) ** 2, axis=(1,2)) / y_hat.shape[1]

    @staticmethod
    def eval_acc(best_preds, y, problem_type):
        if problem_type == ProblemType.CLASS_SINGLE:
            return cp.mean(abs(y - best_preds) < .5)
        elif problem_type == ProblemType.CLASS_ONEHOT:
            return cp.mean(cp.argmax(best_preds, axis=1) == cp.argmax(y, axis=1))
        elif problem_type == ProblemType.REGRESS:
            return cp.mean(abs(y - best_preds) < .5)
        
    class Expansion():
        @staticmethod
        def ID(a, depth, minx=None):
            return cp.repeat(a[:, cp.newaxis, :, :], depth, axis=1)
        
        @staticmethod
        def NID(a, depth, minx=None):
            return cp.repeat(a[:, cp.newaxis, :, :] / depth ** .5, depth, axis=1)
        
        @staticmethod
        def F(a, depth, minx=1):
            ax = a[:, cp.newaxis, :, :]
            return cp.concatenate((cp.repeat(ax, depth - 2, axis=1), cp.sqrt(ax + minx), cp.log(ax + minx + 1)), axis=1)
        
    class Aggregation():
        @staticmethod
        def MEAN(a):
            return cp.mean(a, axis=1)
        
        @staticmethod
        def L2(a):
            return cp.linalg.norm(a, axis=1)
        
class DAGNN:
    
    def __init__(self, inputs, hidden, outputs, problem_type, pop_size, dropout, log_file=None):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.total = inputs + hidden + outputs
        self.problem_type = problem_type
        self.pop_size = pop_size
        self.dropout = 1 - dropout
        self.log_file = log_file
        self._initialize_nets()
        
    def train(self, x_train, y_train, x_test, y_test, iterations):
        train_in = cp.tile(cp.concatenate((x_train, cp.zeros(shape=(x_train.shape[0], self.hidden + self.outputs))), axis=1), (self.pop_size, 1, 1))
        test_in = cp.tile(cp.concatenate((x_test, cp.zeros(shape=(x_test.shape[0], self.hidden + self.outputs))), axis=1), (self.pop_size, 1, 1))
        for i in range(iterations):
            start_time = time.time()
            y_hat = self._forward_prop(train_in, True)
            fitness, train_loss, train_acc = self.evaluate(y_hat, y_train, self.problem_type)
            self._evolve(fitness)
            test_loss, test_acc = self.evaluate(self._forward_prop(test_in, False), y_test, self.problem_type)[1:]
            print('[GENERATION %i/%i]\n\tTIME: %.2f seconds | TRAIN Loss: %.4f Acc: %.4f | TEST Loss: %.4f Acc: %.4f\n' % (i + 1, iterations, time.time() - start_time, train_loss, train_acc, test_loss, test_acc))
            if self.log_file:
                with open(self.log_file, 'a+') as file:
                    file.write(' '.join([str(s) for s in [i + 1, iterations, time.time() - start_time, train_loss, train_acc, test_loss, test_acc]]) + '\n')
        return test_acc
    
    def _initialize_nets(self):
        self.weights = cp.random.normal(size=(self.pop_size, self.total, self.total), dtype='single') * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, 1, 1)) * (cp.random.uniform(size=(self.pop_size, self.total, self.total), dtype='single') < .5)
        self.weights[:, :, self.inputs:] *= cp.sqrt(4 / cp.minimum(cp.arange(self.inputs, self.total), self.inputs + self.hidden))
        self.weights[:, :self.inputs, :self.inputs] = 0
        self.weights[:, -self.outputs:, -self.outputs:] = cp.tile(cp.diag(cp.diag(cp.ones(shape=(self.outputs, self.outputs), dtype='bool_'))), (self.pop_size, 1, 1))
        self.biases = cp.random.normal(size=(self.pop_size, 1, self.hidden + self.outputs), dtype='single') * .5
        
    def _forward_prop(self, inp, training):
        if training:
            mask = self._generate_dropout_mask(inp.shape[1])
        fprop = DAGNN.hadadot(inp, self.weights)
        for i in range(self.inputs, self.inputs + self.hidden):
            wedge = DAGNN.relu(fprop[:, :, [i]] + self.biases[:, :, [i - self.inputs]]) * (mask[:, [i - self.inputs]] if training else 1)
            fprop[:, :, i] = 0
            fprop += DAGNN.hadadot(wedge, self.weights[:, [i], :]) / self.dropout
        if self.problem_type == ProblemType.CLASS_SINGLE:
            return DAGNN.sigmoid(fprop[:, :, -self.outputs:] + self.biases[:, :, -self.outputs:])
        elif self.problem_type == ProblemType.CLASS_ONEHOT:
            return DAGNN.softmax(fprop[:, :, -self.outputs:] + self.biases[:, :, -self.outputs:])
        elif self.problem_type == ProblemType.REGRESS:
            return fprop[:, :, -self.outputs:] + self.biases[:, :, -self.outputs:]
    
    def _generate_dropout_mask(self, dim):
        return (cp.random.uniform(size=(dim, self.hidden)) < self.dropout) / self.dropout
    
    def _evolve(self, fitness):
        selected = self._select(fitness)
        self._crossover(selected)
        self._mutate()
        
    def _select(self, fitness):
        order = cp.argsort(fitness)
        return order[(cp.random.uniform(size=(2, self.pop_size)) ** 2 * self.pop_size).astype(int)]
        
    def _crossover(self, selected):
        point = cp.random.randint(1, self.total-1).item()
        self.weights = cp.concatenate((self.weights[selected[0, :], :point, :], self.weights[selected[1, :], point:, :]), axis=1)
        self.biases = self.biases[selected[1, :], :, :] if point <= self.inputs else cp.concatenate((self.biases[selected[0, :], :, :point - self.inputs], self.biases[selected[1, :], :, point - self.inputs:]), axis=2)

    def _mutate(self):
        weight_mask = cp.random.normal(size=(self.weights.shape), dtype='single') * .05 * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, 1, 1)) * (cp.random.uniform(size=(self.weights.shape), dtype='single') < .2) * (self.weights != 0)
        new_weights_mask = cp.random.normal(size=(self.weights.shape), dtype='single') * .05 * cp.tile(cp.triu(cp.ones(shape=(self.total, self.total), dtype='bool_'), 1), (self.pop_size, 1, 1)) * (cp.random.uniform(size=(self.weights.shape), dtype='single') < .05) * (self.weights == 0)
        self.weights += weight_mask + new_weights_mask
        self.weights *= (cp.random.uniform(size=self.weights.shape, dtype='single') < 1)
        self.weights[:, :self.inputs, :self.inputs] = 0
        self.weights[:, -self.outputs:, -self.outputs:] = cp.tile(cp.diag(cp.diag(cp.ones(shape=(self.outputs, self.outputs), dtype='bool_'))), (self.pop_size, 1, 1))
        self.biases += cp.random.normal(size=self.biases.shape, dtype='single') * .05 * (cp.random.normal(size=self.biases.shape, dtype='single') < .2)
    
    @staticmethod
    def relu(a):
        return cp.maximum(a, 0)
    
    @staticmethod
    def hadadot(a, b):
        return a @ b
    
    @staticmethod
    def softmax(a):
        return cp.exp(a) / cp.sum(cp.exp(a), axis=2, keepdims=True)
    
    @staticmethod
    def sigmoid(a):
        return 1 / (1 + cp.exp(-a))
    
    @staticmethod
    def evaluate(y_hat, y, problem_type):
        fitness = DAGNN.eval_fitness(y_hat, y, problem_type)
        order = cp.argsort(fitness)
        best_loss = fitness[order[0]]
        best_preds = y_hat[order[0]]
        best_acc = DAGNN.eval_acc(best_preds, y, problem_type)
        return fitness, best_loss, best_acc
    
    @staticmethod
    def eval_fitness(y_hat, y, problem_type):
        if problem_type in [ProblemType.CLASS_SINGLE, ProblemType.CLASS_ONEHOT]:
            return DAGNN.crossentropy(y_hat, y)
        elif problem_type == ProblemType.REGRESS:
            return DAGNN.mse(y_hat, y)
    
    @staticmethod
    def crossentropy(y_hat, y):
        return -1 / y_hat.shape[1] * cp.sum(y * cp.log(y_hat + 1e-5) + (1 - y) * cp.log(1 - y_hat + 1e-5), axis=(1,2))
    
    @staticmethod
    def mse(y_hat, y):
        return cp.sum((y_hat - y) ** 2, axis=(1,2)) / y_hat.shape[1]

    @staticmethod
    def eval_acc(best_preds, y, problem_type):
        if problem_type == ProblemType.CLASS_SINGLE:
            return cp.mean(abs(y - best_preds) < .5)
        elif problem_type == ProblemType.CLASS_ONEHOT:
            return cp.mean(cp.argmax(best_preds, axis=1) == cp.argmax(y, axis=1))
        elif problem_type == ProblemType.REGRESS:
            return cp.mean(abs(y - best_preds) < .5)