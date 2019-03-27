# for observing data
import matplotlib.pyplot as plt


# parsing mnist
import numpy as np

from snn.util import convert_kernel
from snn.util import MNIST


if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    mnist = MNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
    labels = mnist.load_labels()
    images = mnist.load_images()

    magic, size, rows, cols = mnist.get_image_meta()
    #

    # (train_labels, train_images) = mnist.load_data()

    # pulls an image at location
    image_at = 5
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]
    # normalize data set
    image = np.reshape(image, (28, 28))

    snn = SNN(image.shape[0] * image.shape[1], kernels, 9)
    snn.set_model()
    guess = snn.guess(image)

    output = np.zeros((26, 26))

    kernel = "0 0 1.6 0 1.6 0 0 0 1.6"
    kernel = convert_kernel(kernel)

    kernel_width = kernel.shape[0]

    for x in range(0, image.shape[1] - 2):
        for y in range(0, image.shape[0] - 2):
            print(y)
            output[x, y] = (
                kernel * image[x : x + kernel_width, y : y + kernel_width]
            ).sum()

    plt.title("Convolution")
    plt.imshow(output, interpolation="nearest")
    plt.show()
    plt.title("Original")
    plt.imshow(image, interpolation="nearest")
    plt.show()
