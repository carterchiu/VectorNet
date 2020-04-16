''' Demonstrates usage of the DAGNN module to train a vector-weighted DAG-NN on toy data '''

from DAGNN import VecDAGNN, ProblemType
import cupy as cp

# generate some toy data
X_train = cp.random.uniform(-1, 1, (1000,5))
y_train = (X_train.sum(axis=1, keepdims=True) > 0).astype('int')
X_test = cp.random.uniform(-1, 1, (100,5))
y_test = (X_test.sum(axis=1, keepdims=True) > 0).astype('int')

m = VecDAGNN(5, 2, 1, 3, VecDAGNN.Expansion.ID, VecDAGNN.Aggregation.MEAN, ProblemType.CLASS_SINGLE, 100, 0)
print('TEST ACC:', m.train(X_train, y_train, X_test, y_test, 500).item())