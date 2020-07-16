"""A collection of functions for loading the MNIST database of handwritten digits.

The specifications for the file formats as well as the files themselves can be found in the link below.

http://yann.lecun.com/exdb/mnist/
"""

import numpy as np

class FileFormatException(Exception):
    pass

def mnist_label_file(filename):
    label_file_magic_number = 0x00000801
    with open(filename, "rb") as file:
        magic_number = int.from_bytes(file.read(4), byteorder='big', signed=False)
        if magic_number != label_file_magic_number:
            raise FileFormatException(f"invalid magic number {magic_number} should be {label_file_magic_number}")
        number_of_items = int.from_bytes(file.read(4), byteorder='big', signed=False)
        return np.fromfile(file, dtype=np.uint8, count=number_of_items)

def mnist_image_file(filename):
    image_file_magic_number = 0x00000803
    with open(filename, "rb") as file:
        magic_number = int.from_bytes(file.read(4), byteorder='big', signed=False)
        if magic_number != image_file_magic_number:
            raise FileFormatException(f"invalid magic number {magic_number} should be {image_file_magic_number}")
        number_of_images = int.from_bytes(file.read(4), byteorder='big', signed=False)
        number_of_rows = int.from_bytes(file.read(4), byteorder='big', signed=False)
        number_of_columns = int.from_bytes(file.read(4), byteorder='big', signed=False)
        pixels = number_of_images * number_of_rows * number_of_columns
        shape = (number_of_images, number_of_rows, number_of_columns)
        return np.fromfile(file, dtype=np.uint8, count=pixels).reshape(shape)
