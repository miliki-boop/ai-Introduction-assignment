import struct
import numpy as np

def decode_idx3_ubyte(idx3_ubyte_file):
    '''
        解析图片文件,每一个像素值0-255,未归一化
        解析train-images,返回(60000,28,28) ndarray dtype = float64
        解析test-images,返回(10000,28,28) ndarray  dtype = float64
    '''
    with open(idx3_ubyte_file, 'rb') as f:
        print('解析文件：', idx3_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'    # 以大端法读取4个 unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，图片数：{}'.format(magic_number, num_images))
    print(num_rows,num_cols)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    '''
        解析标签文件
        解析train-labels,返回60000个元素的列表
        解析test-labels,返回10000个元素的列表
    '''
    with open(idx1_ubyte_file, 'rb') as f:
        print('解析文件：', idx1_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'  # 以大端法读取两个 unsinged int32
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，标签数：{}'.format(magic_number, label_num))
    offset += struct.calcsize(fmt_header)
    labels = []

    fmt_label = '>B'    # 每次读取一个 byte
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return labels

# test_images = './Mnist/Mnist/test-images.idx3-ubyte'
# test_labels = './Mnist/Mnist/test-labels.idx1-ubyte'
# train_images = './Mnist/Mnist/train-images.idx3-ubyte'
# train_labels = './Mnist/Mnist/train-labels.idx1-ubyte'

# print(decode_idx1_ubyte(test_labels))
# print(decode_idx3_ubyte(test_images)[1])