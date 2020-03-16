from keras.datasets import mnist

IMG_WIDTH, IMG_HEIGHT = 28, 28

def scale_mnist_data(mnist_data):
    return mnist_data / 255
    
def get_normalized_data():
    (train_data, y_train), (test_data, y_test) = mnist.load_data()
    normalized_train = scale_mnist_data(mnist_data=train_data)
    final_train = normalized_train.reshape(-1,
                                           IMG_WIDTH * IMG_HEIGHT)  # turn shape (60000, 28, 28) into flat (60000, 784)

    normalized_test = scale_mnist_data(mnist_data=test_data)
    final_test = normalized_test.reshape(-1, IMG_WIDTH * IMG_HEIGHT)
    return (final_train, y_train), (final_test, y_test)
