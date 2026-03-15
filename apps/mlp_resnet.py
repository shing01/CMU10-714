import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_path = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )

    return nn.Sequential(
        nn.Residual(main_path),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0.0
    total_error = 0.0
    num_samples = 0.0

    for X, y in dataloader:
        if opt:
            opt.reset_grad()
        logits = model(X)
        loss = loss_fn(logits, y)

        if opt:
            loss.backward()
            opt.step()

        batch_size = X.shape[0]
        total_loss += loss.numpy() * batch_size
        preds = np.argmax(logits.numpy(), axis=1)
        total_error += np.sum(preds != y.numpy())
        num_samples += batch_size

    return total_error / num_samples, total_loss / num_samples
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        image_filename=f"{data_dir}/train-images-idx3-ubyte.gz",
        label_filename=f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
        image_filename=f"{data_dir}/t10k-images-idx3-ubyte.gz",
        label_filename=f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    train_loader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPResNet(dim=784, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
