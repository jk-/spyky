import numpy as np

import struct
from array import array
import matplotlib.pyplot as plt

from snn.util import convert_kernel

kernel = "0 0 1.6 \
0 1.6 0 \
0 0 1.6"


# k = convert_kernel(kernel)
# print(k)
# quit()
with open("data/t10k-labels-idx1-ubyte", "rb") as file:
    # magic, number of items
    magic, size = struct.unpack(">II", file.read(8))
    if magic != 2049:
        raise ValueError(
            "Magic number mismatch, expected 2049," "got {}".format(magic)
        )

    labels = array("B", file.read())


with open("data/t10k-images-idx3-ubyte", "rb") as file:
    # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    if magic != 2051:
        raise ValueError(
            "Magic number mismatch, expected 2051," "got {}".format(magic)
        )

    images = array("B", file.read())

# pulls an image at location
image_at = 5
image = images[
    (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
]
# normalize data set
# image = [x / 255 for x in image]
image = np.reshape(image, (28, 28))

padded_picture = np.zeros((image.shape[0] + 2, image.shape[1] + 2))

padded_picture[1:-1, 1:-1] = image

output = np.zeros_like(image)

kernel = convert_kernel(kernel)

for x in range(image.shape[1]):
    for y in range(image.shape[0]):
        output[y, x] = (kernel * padded_picture[y : y + 3, x : x + 3]).sum()

plt.title("Convolution")
plt.imshow(output, interpolation="nearest")
plt.show()
plt.title("Original")
plt.imshow(image, interpolation="nearest")
plt.show()


#
# # Because we need to account for the edges of the image we need
# # to pad the picture with zeros
# padded_picture = np.zeros((picture.shape[0] + 2, picture.shape[1] + 2, 3))
# #
# # # inject the picture inside the padded_picture
# # # python <3
# padded_picture[1:-1, 1:-1] = picture


# this moves the kernel over each section of the picture
# and gets the summized dot product
# for x in range(picture.shape[1]):
#     for y in range(picture.shape[0]):
#         output[y, x] = (kernel * padded_picture[y : y + 3, x : x + 3]).sum()
#
#
# plt.imshow(image, interpolation="nearest", cmap="gray")
# plt.show()
