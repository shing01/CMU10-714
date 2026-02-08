import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 由于硬件不同，在计算 np.linalg.norm 时可能会有微小的数值差异，因此我们将 rtol 从 1e-6 放宽到 1e-5
    with gzip.open(image_filename, 'rb') as f:
        # 直接读取整个文件，利用 np.frombuffer 的 offset 参数跳过文件头
        data = f.read()
        # 前 16 个字节是 Header（4字节魔数 + 4字节样本数 + 4字节行数 + 4字节列数）
        # struct.unpack_from 可以从指定位置解析
        magic, num_images, rows, cols = struct.unpack_from('>IIII', data, 0)
        assert magic == 2051, "Invalid magic number in image file: {}".format(magic)
        X = np.frombuffer(data, dtype=np.uint8, offset=16)
        # Reshape: (N, H * W) -> (60000, 784)
        X = X.reshape(num_images, rows * cols)
        X = X.astype(np.float32) / np.float32(255.0)

    with gzip.open(label_filename, 'rb') as f:
        data = f.read()
        # 前 8 个字节是 Header（4字节魔数 + 4字节样本数）
        magic, num_labels = struct.unpack_from('>II', data, 0)
        assert magic == 2049, "Invalid magic number in label file: {}".format(magic)
        y = np.frombuffer(data, dtype=np.uint8, offset=8)

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # 计算所有元素的指数 exp(Z)
    exp_Z = np.exp(Z)
    # 每一行的指数之和
    sum_exp_Z = np.sum(exp_Z, axis=1)
    # 取对数
    log_sum_exp_Z = np.log(sum_exp_Z)
    # 取出真实标签对应的 Logit 值 Z[i, y[i]]
    batch_size = Z.shape[0]
    Z_y = Z[np.arange(batch_size), y]
    # 计算 Loss: LogSumExp - TrueClassLogit
    loss = log_sum_exp_Z - Z_y

    return np.mean(loss)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    for i in range(0, num_examples, batch):
        # 确定当前 Batch 的切片范围
        end = min(i + batch, num_examples)
        X_batch = X[i:end]
        y_batch = y[i:end]
        current_batch_size = end - i
        # 前向传播
        Z = np.dot(X_batch, theta)
        # Softmax
        exp_Z = np.exp(Z)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True) # keepdims=True 保持二维，方便后续广播除法
        P = exp_Z / sum_exp_Z
        # 计算梯度
        Z_grad = P
        Z_grad[np.arange(current_batch_size), y_batch] -= 1
        grad = np.dot(X_batch.T, Z_grad) / current_batch_size
        # 更新参数
        theta -= lr * grad


    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    for i in range(0, num_examples, batch):
        end = min(i + batch, num_examples)
        X_batch = X[i:end]
        y_batch = y[i:end]
        current_batch_size = end - i
        # 前向传播
        # Layer 1: Linear + ReLU
        Z1 = np.dot(X_batch, W1)
        A1 = np.maximum(Z1, 0)
        # Layer 2: Linear (Logits)
        Z2 = np.dot(A1, W2)
        # Softmax
        exp_Z2 = np.exp(Z2)
        sum_exp_Z2 = np.sum(exp_Z2, axis=1, keepdims=True)
        P = exp_Z2 / sum_exp_Z2

        # 反向传播
        # 输出层梯度 (G2 = P - y_onehot)
        G2 = P
        G2[np.arange(current_batch_size), y_batch] -= 1
        # 第二层权重梯度 (Grad W2)
        grad_W2 = np.dot(A1.T, G2) / current_batch_size
        # 传递到隐藏层 (G1)
        G1 = np.dot(G2, W2.T)
        # ReLU求导
        G1[Z1 <= 0] = 0
        # 第一层权重梯度 (Grad W1)
        grad_W1 = np.dot(X_batch.T, G1) / current_batch_size
        # 更新参数
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
