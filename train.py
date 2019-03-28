# for observing data
import matplotlib.pyplot as plt


# parsing mnist
import numpy as np

from snn.util import convert_kernel
from snn.util import MNIST
from snn import SNN


if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    mnist = MNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
    labels = mnist.load_labels()
    images = mnist.load_images()

    magic, size, rows, cols = mnist.get_image_meta()

    kernels = [
        "0 1 0 0 1 0 0 1 0",
        "0 0 0 1 1 1 0 0 0",
        "1 0 0 0 1 0 0 0 1",
        "0 0 1 0 1 0 1 0 0",
        "0 1 0 0 1 1 0 0 0",
        "0 0 0 1 1 0 0 1 0",
        "0 1 0 1 1 0 0 0 0",
        "0 0 0 0 1 1 0 1 0",
        "1 0 1 0 1 0 0 0 0",
        "1 0 0 0 1 0 1 0 0",
        "0 0 0 0 1 0 1 0 1",
        "0 0 1 0 1 0 0 0 1",
    ]
    snn = SNN(28 * 28, kernels, 10)

    image_at = 1
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]

    # normalize data set
    image = [x / 255 for x in image]
    image = np.reshape(image, (28, 28))
    snn.set_input(image)
    snn.print_neurons()

    # guess = snn.guess(image)

    # (train_labels, train_images) = mnist.load_data()

    # pulls an image at location
    # image_at = 1
    # image = images[
    #     (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    # ]
    # # normalize data set
    # image = np.reshape(image, (28, 28))
    #
    # output = np.zeros((26, 26))
    #
    # kernel = ""
    # kernel = convert_kernel(kernel)
    #
    # kernel_width = kernel.shape[0]
    #
    # for x in range(0, image.shape[1] - 2):
    #     for y in range(0, image.shape[0] - 2):
    #         output[x, y] = (
    #             kernel * image[x : x + kernel_width, y : y + kernel_width]
    #         ).sum()
    #
    # print(output.shape)
    #
    # plt.title("Convolution")
    # plt.imshow(output, interpolation="nearest")
    # plt.show()
    # plt.title("Original")
    # plt.imshow(image, interpolation="nearest")
    # plt.show()
