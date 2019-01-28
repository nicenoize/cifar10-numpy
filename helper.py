import numpy as np
import os, pickle, math
import matplotlib.pyplot as plt

#########################################################################
#
#        Helper functions
#
#########################################################################


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        datadict = u.load()
        # datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def prepare_CIFAR10_images(num_training=49000, num_validation=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)

    return X_train, y_train, X_val, y_val


#########################################################################
#
#       Visualization Helper functions
#
#########################################################################

def plot_loss(stats):
    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.plot(stats['loss_val_history'], 'r')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss history (red is testing set)')
    plt.show()


def plot_accuracy(stats):
    plt.plot(stats['train_acc_history'])
    plt.axhline(y=0.47, color='0.75', linestyle='dotted')
    plt.axhline(y=0.5, color='0.75', linestyle='dashdot')
    plt.axhline(y=0.55, color='0.75', linestyle='dashed')
    plt.plot(stats['val_acc_history'], 'r')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy history (red is testing set)')
    plt.show()


def visualize_grid_of_images(Xs, ubound=255.0, padding=1):
    """
    Visualisierung einer Gewichtungsmatrix
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def plot_net_weights(net):
    W1 = net.W1
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid_of_images(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
