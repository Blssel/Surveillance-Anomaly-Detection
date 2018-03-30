import struct
import numpy as np
import argparse


def read_binary_fc(path):
    # from github https://github.com/facebook/C3D/issues/22
    # 5 32-bit integers: num, chanel, length, height, width. (record the size of the blob)
    # Then followed by the data of (num * channel * length * height * width) each data is a 32-bit float in row order.
    with open(path, 'rb') as f:

        elements = 5                #: num, chanel, length, height, width
        element_size_byte = 4       # 32bit = 4byte
        total_header_size = elements*element_size_byte

        # The width and height are 4 bytes each, so read 8 bytes to get both of them
        header_bytes = f.read(total_header_size)

        # Here, we decode the byte array from the last step.
        header = struct.unpack('i' * elements, header_bytes)

        data_size = header[0]*header[1]*header[2]*header[3]*header[4]

        data_bytes = f.read(element_size_byte*data_size)

        data = struct.unpack('f' * data_size, data_bytes)

        return header, np.array(data)


def main(args):

    header, data =  read_binary_fc(args.path)
    print "info inside file: ", args.path
    print "HEADER: \tnum, chanel, length, height, width.\n\t\t", header
    print "DATA:   \tshape:\t", data.shape
    print "\t\t", data
    print "\t\tsum:  ", data.sum()


def get_parser():
    parser = argparse.ArgumentParser(description="Read C3D binary feature vector")
    parser.add_argument("-path", help="Path to fc binary file, default to test file", default='converter_test/binary/000000.fc7-1')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
