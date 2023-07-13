import numpy as np
import os
import cv2
import sys
import json
from labelme import utils

""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" --------------------------------------------------------------Global Variable Definitions------------------------------------------------------------------"""
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
FNC3 = 96 + 32
FNC2 = 97 + 32
SHIFT = 98 + 32
CODE_C = 99 + 32
CODE_B = 100 + 32
FNC4_B = 100 + 32
CODE_A = 101 + 32
FNC4_A = 101 + 32
FNC1 = 102 + 32
START_A = 103 + 32
START_B = 104 + 32
START_C = 105 + 32
STOP = 106 + 32
c128_bp = [
    [' ', ' ', 0, 2, 1, 2, 2, 2, 2],
    ['!', '!', 1, 2, 2, 2, 1, 2, 2],
    ['"', '"', 2, 2, 2, 2, 2, 2, 1],
    ['#', '#', 3, 1, 2, 1, 2, 2, 3],
    ['$', '$', 4, 1, 2, 1, 3, 2, 2],
    ['%', '%', 5, 1, 3, 1, 2, 2, 2],
    ['&', '&', 6, 1, 2, 2, 2, 1, 3],
    ['\'', '\'', 7, 1, 2, 2, 3, 1, 2],
    ['(', '(', 8, 1, 3, 2, 2, 1, 2],
    [')', ')', 9, 2, 2, 1, 2, 1, 3],
    ['*', '*', 10, 2, 2, 1, 3, 1, 2],
    ['+', '+', 11, 2, 3, 1, 2, 1, 2],
    [',', ',', 12, 1, 1, 2, 2, 3, 2],
    ['-', '-', 13, 1, 2, 2, 1, 3, 2],
    ['.', '.', 14, 1, 2, 2, 2, 3, 1],
    ['/', '/', 15, 1, 1, 3, 2, 2, 2],
    ['0', '0', 16, 1, 2, 3, 1, 2, 2],
    ['1', '1', 17, 1, 2, 3, 2, 2, 1],
    ['2', '2', 18, 2, 2, 3, 2, 1, 1],
    ['3', '3', 19, 2, 2, 1, 1, 3, 2],
    ['4', '4', 20, 2, 2, 1, 2, 3, 1],
    ['5', '5', 21, 2, 1, 3, 2, 1, 2],
    ['6', '6', 22, 2, 2, 3, 1, 1, 2],
    ['7', '7', 23, 3, 1, 2, 1, 3, 1],
    ['8', '8', 24, 3, 1, 1, 2, 2, 2],
    ['9', '9', 25, 3, 2, 1, 1, 2, 2],
    [':', ':', 26, 3, 2, 1, 2, 2, 1],
    [';', ';', 27, 3, 1, 2, 2, 1, 2],
    ['<', '<', 28, 3, 2, 2, 1, 1, 2],
    ['=', '=', 29, 3, 2, 2, 2, 1, 1],
    ['>', '>', 30, 2, 1, 2, 1, 2, 3],
    ['?', '?', 31, 2, 1, 2, 3, 2, 1],
    ['@', '@', 32, 2, 3, 2, 1, 2, 1],
    ['A', 'A', 33, 1, 1, 1, 3, 2, 3],
    ['B', 'B', 34, 1, 3, 1, 1, 2, 3],
    ['C', 'C', 35, 1, 3, 1, 3, 2, 1],
    ['D', 'D', 36, 1, 1, 2, 3, 1, 3],
    ['E', 'E', 37, 1, 3, 2, 1, 1, 3],
    ['F', 'F', 38, 1, 3, 2, 3, 1, 1],
    ['G', 'G', 39, 2, 1, 1, 3, 1, 3],
    ['H', 'H', 40, 2, 3, 1, 1, 1, 3],
    ['I', 'I', 41, 2, 3, 1, 3, 1, 1],
    ['J', 'J', 42, 1, 1, 2, 1, 3, 3],
    ['K', 'K', 43, 1, 1, 2, 3, 3, 1],
    ['L', 'L', 44, 1, 3, 2, 1, 3, 1],
    ['M', 'M', 45, 1, 1, 3, 1, 2, 3],
    ['N', 'N', 46, 1, 1, 3, 3, 2, 1],
    ['O', 'O', 47, 1, 3, 3, 1, 2, 1],
    ['P', 'P', 48, 3, 1, 3, 1, 2, 1],
    ['Q', 'Q', 49, 2, 1, 1, 3, 3, 1],
    ['R', 'R', 50, 2, 3, 1, 1, 3, 1],
    ['S', 'S', 51, 2, 1, 3, 1, 1, 3],
    ['T', 'T', 52, 2, 1, 3, 3, 1, 1],
    ['U', 'U', 53, 2, 1, 3, 1, 3, 1],
    ['V', 'V', 54, 3, 1, 1, 1, 2, 3],
    ['W', 'W', 55, 3, 1, 1, 3, 2, 1],
    ['X', 'X', 56, 3, 3, 1, 1, 2, 1],
    ['Y', 'Y', 57, 3, 1, 2, 1, 1, 3],
    ['Z', 'Z', 58, 3, 1, 2, 3, 1, 1],
    ['[', '[', 59, 3, 3, 2, 1, 1, 1],
    ['\\', '\\', 60, 3, 1, 4, 1, 1, 1],
    [']', ']', 61, 2, 2, 1, 4, 1, 1],
    ['^', '^', 62, 4, 3, 1, 1, 1, 1],
    ['_', '_', 63, 1, 1, 1, 2, 2, 4],
    ['NUL', '`', 64, 1, 1, 1, 4, 2, 2],
    ['SOH', 'a', 65, 1, 2, 1, 1, 2, 4],
    ['STX', 'b', 66, 1, 2, 1, 4, 2, 1],
    ['ETX', 'c', 67, 1, 4, 1, 1, 2, 2],
    ['EOT', 'd', 68, 1, 4, 1, 2, 2, 1],
    ['ENQ', 'e', 69, 1, 1, 2, 2, 1, 4],
    ['ACK', 'f', 70, 1, 1, 2, 4, 1, 2],
    ['BEL', 'g', 71, 1, 2, 2, 1, 1, 4],
    ['BS', 'h', 72, 1, 2, 2, 4, 1, 1],
    ['HT', 'i', 73, 1, 4, 2, 1, 1, 2],
    ['LF', 'j', 74, 1, 4, 2, 2, 1, 1],
    ['VT', 'k', 75, 2, 4, 1, 2, 1, 1],
    ['FF', 'l', 76, 2, 2, 1, 1, 1, 4],
    ['CR', 'm', 77, 4, 1, 3, 1, 1, 1],
    ['SO', 'n', 78, 2, 4, 1, 1, 1, 2],
    ['SI', 'o', 79, 1, 3, 4, 1, 1, 1],
    ['DLE', 'p', 80, 1, 1, 1, 2, 4, 2],
    ['DC1', 'q', 81, 1, 2, 1, 1, 4, 2],
    ['DC2', 'r', 82, 1, 2, 1, 2, 4, 1],
    ['DC3', 's', 83, 1, 1, 4, 2, 1, 2],
    ['DC4', 't', 84, 1, 2, 4, 1, 1, 2],
    ['NAK', 'u', 85, 1, 2, 4, 2, 1, 1],
    ['SYN', 'v', 86, 4, 1, 1, 2, 1, 2],
    ['ETB', 'w', 87, 4, 2, 1, 1, 1, 2],
    ['CAN', 'x', 88, 4, 2, 1, 2, 1, 1],
    ['EM', 'y', 89, 2, 1, 2, 1, 4, 1],
    ['SUB', 'z', 90, 2, 1, 4, 1, 2, 1],
    ['ESC', '{', 91, 4, 1, 2, 1, 2, 1],
    ['FS', '|', 92, 1, 1, 1, 1, 4, 3],
    ['GS', '}', 93, 1, 1, 1, 3, 4, 1],
    ['RS', '~', 94, 1, 3, 1, 1, 4, 1],
    ['US', 'DEL', 95, 1, 1, 4, 1, 1, 3],
    ['FNC3', 'FNC3', 96, 1, 1, 4, 3, 1, 1],
    ['FNC2', 'FNC2', 97, 4, 1, 1, 1, 1, 3],
    ['SHIFT', 'SHIFT', 98, 4, 1, 1, 3, 1, 1],
    ['CODE_C', 'CODE_C', 99, 1, 1, 3, 1, 4, 1],
    ['CODE_B', 'FNC4_B', CODE_B, 1, 1, 4, 1, 3, 1],
    ['FNC4_A', 'CODE_A', CODE_A, 3, 1, 1, 1, 4, 1],
    ['FNC1', 'FNC1', FNC1, 4, 1, 1, 1, 3, 1],
    ['START_A', 'START_A', START_A, 2, 1, 1, 4, 1, 2],
    ['START_B', 'START_B', START_B, 2, 1, 1, 2, 1, 4],
    ['START_C', 'START_C', START_C, 2, 1, 1, 2, 3, 2]]

c128_stop = [2, 3, 3, 1, 1, 1, 2]

color2bit_map = {'red': '00',
                 'green': '01',
                 'blue': '10',
                 'black': '11'}
bits2value_map = {'00': 1,
                  '01': 2,
                  '10': 3,
                  '11': 4}


"""" ----------Utils Function---------- """


def code2string(code):
    """将list of code转为字符串"""
    # code为一维list
    code = [str(value) for value in code]
    code_string = ''
    for value in code:
        code_string += value
    return code_string.strip()


def get_c128AB_value_code():
    """返回code128A/B的字典"""
    value2code = [code[3:] for code in c128_bp]
    str_code2value = {}

    for i, code in enumerate(value2code):
        code_string = code2string(code)
        str_code2value[f'{code_string}'] = i

    # 添加stop符号
    value2code.append(c128_stop)
    str_code2value[f'2331112'] = len(c128_bp)
    return value2code, str_code2value


def get_c128AB_value_char():
    """返回code128A/B的字典"""
    value2charA = [code[0] for code in c128_bp]
    value2charB = [code[1] for code in c128_bp]
    charA2value, charB2value = {}, {}
    for i, char in enumerate(value2charA):
        charA2value[char] = i
    for i, char in enumerate(value2charB):
        charB2value[char] = i

    value2charA.append('STOP')
    value2charB.append('STOP')
    charA2value['STOP'] = len(c128_bp)
    charB2value['STOP'] = len(c128_bp)

    return value2charA, value2charB, charA2value, charB2value


def get_c128AB():
    """得到code128的各个转换list和dict"""
    value2code, str_code2value = get_c128AB_value_code()
    value2charA, value2charB, charA2value, charB2value = get_c128AB_value_char()
    return value2code, str_code2value, value2charA, value2charB, charA2value, charB2value


def get_c128_type(barcode):
    if code2string(barcode[0]) == "211412":
        return 'A'
    if code2string(barcode[0]) == "211214":
        return 'B'


""" ----------Conversion Function---------- """


def bits2values(bitstreams):
    """将比特流转为数值"""
    if type(bitstreams) == list:
        bitstreams = np.array([np.array(bitstream, dtype=np.int8)
                               for bitstream in bitstreams])

    # bitstream shape: (num_chars, 2 * 6)
    bits2value_map = {'00': 1,
                      '01': 2,
                      '10': 3,
                      '11': 4}

    values = []
    for row in bitstreams:
        value = []
        for i, bit in enumerate(row):
            value.append(bit)
            if (i + 1) % 2 == 0:
                bits = str(value[0]) + str(value[1])
                values.append(bits2value_map[bits])
                value.clear()
    return np.array(values, dtype=np.int8).reshape(bitstreams.shape[0], -1)


def values2bits(values):
    """将数值转为bit流"""
    # bits.shape: (num_chars, 6)
    value_map = [None, '0,0,', '0,1,', '1,0,', '1,1,']
    bit_streams = []

    for row in values:
        bit_stream = ' '
        for value in row:
            bit_stream += value_map[value]
        bit_streams.append(bit_stream.strip().split(',')[:-1])

    for row, row_values in enumerate(bit_streams):
        for col, value in enumerate(row_values):
            bit_streams[row][col] = int(value)

    return np.array(bit_streams, dtype=np.int8)


def barcode2chars(barcode, c128AB, code_type='A'):
    """将barcode矩阵转为characters"""
    # barcode.shape == num_char, 6
    value2code, str_code2value, value2charA, value2charB, charA2value, charB2value = c128AB
    if code_type == 'A':
        chars = [value2charA[str_code2value[code2string(code)]] for code in barcode
                 if code2string(code) in str_code2value.keys()]
    elif code_type == 'B':
        chars = [value2charB[str_code2value[code2string(code)]] for code in barcode
                 if code2string(code) in str_code2value.keys()]
    return chars


def chars2barcode(chars, c128AB, code_type='A'):
    """将characters矩阵转为barcode"""
    # chars.shape == num_char,
    value2code, str_code2value, value2charA, value2charB, charA2value, charB2value = c128AB
    if code_type == 'A':
        barcode = [value2code[charA2value[charA]] for charA in chars
                   if charA in charA2value]
    elif code_type == 'B':
        barcode = [value2code[charB2value[charB]] for charB in chars
                   if charB in charB2value]
    return barcode


def bitstream2color(bitstreams):
    """将bitstream矩阵转为颜色"""
    # bitstream.shape == num_char, 2 * 6
    bit2color_map = {'00': 'red',
                     '01': 'green',
                     '10': 'blue',
                     '11': 'black'}
    if type(bitstreams) == np.ndarray:
        bitstreams = bitstreams.tolist()
    colors = []
    for bitstream in bitstreams:
        color = []
        bitstream = code2string(bitstream)
        for i in range(len(bitstream)):
            if (i + 1) % 2 == 0:
                color.append(bit2color_map[bitstream[i - 1] + bitstream[i]])
        colors.append(color)
    return colors


def color2bitstream(color_matrix):
    """将颜色矩阵转为bitstream"""

    # colors.shape == num_color,

    def string2code(string):
        return [int(i) for i in string.strip()]

    color2bit_map = {'red': '00',
                     'green': '01',
                     'blue': '10',
                     'black': '11'}

    bitstreams = []
    for colors in color_matrix:
        string = ''
        for color in colors:
            string += color2bit_map[color]
        bitstreams.append(string2code(string))
    return bitstreams


def mod103(in_string, c128AB=get_c128AB()):
    """modulus 103 计算校验位"""
    char_list = in_string
    if in_string[0] in ['START_A', 'START_B']:
        if in_string[0] == 'START_A':
            char2value = c128AB[-2]
            value2char = c128AB[2]
        elif in_string[0] == 'START_B':
            char2value = c128AB[-1]
            value2char = c128AB[3]
        
        value_list = [char2value[char] for char in in_string]
        bits_sum = 0
        for index, value in enumerate(value_list):
            if index == 0:
                bits_sum += value
            bits_sum += value * index

        return value2char[bits_sum % 103]
    
    else:
        print("Warning: Failed to distinguish barcode type...")
        return None


def append_checkbit(in_string, c128AB=get_c128AB()):
    """添加校验位"""
    in_string.append(mod103(in_string, c128AB))
    return in_string


""" 从json文件解码部分 """


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super(MyEncoder, self).default(o)


class JsonUtils:
    """definition of json loader class"""

    def __init__(self, src_json_dir=None, target_json_dir=None):
        self.src_dir = src_json_dir
        self.target_dir = target_json_dir

    # 检查源路径是否存在
    def mkdir(self, src=True):
        if src is True:
            json_path = self.src_dir
        else:
            json_path = self.target_dir
        is_exists = os.path.exists(path=json_path)

        if not is_exists:
            os.mkdir(json_path)
            print('=========')
            print('create path', json_path)
            print('=========')
        return 0

    # 检查目标路径下json是否存在
    def check_files(self, path_list):
        for i in path_list:
            if not os.path.exists(i):
                print('error')
                print(i, 'not exist!!!')
                sys.exit(1)
        print("files exist.")

    # 读取json
    def read_json_files(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            # print(f"read from {path} successfully.")
            return json.load(f)

    # 保存json文件
    def save_json_files(self, json_dict, json_file_name):
        with open(self.target_dir + f"/{json_file_name}", 'w') as write_json:
            write_json.write(json.dumps(json_dict, cls=MyEncoder, indent=2))
        print(f"save {self.target_dir}/{json_file_name} file successfully.")
        print('-' * 100)

    # 从json文件里获取点集坐标
    def get_points_from_json(self, json_file):
        point_list = []
        shapes = json_file['shapes']
        for i in range(len(shapes)):
            for j in range(len(shapes[i]['points'])):
                point_list.append(shapes[i]['points'][j])
        return point_list

    # 将新的点集坐标写回json文件
    def write_points_to_json(self, json_file, aug_points, img_aug, img_name):
        k = 0
        json_dict = {}
        for key, value in json_file.items():
            if key == 'imageHeight':
                json_dict[key] = img_aug.shape[0]
            elif key == 'imageWidth':
                json_dict[key] = img_aug.shape[1]
            elif key == 'imageData':
                json_dict[key] = str(utils.img_arr_to_b64(img_aug), encoding='utf-8')
            elif key == 'imagePath':
                json_dict[key] = ".\\" + img_name[0] + '.' + img_name[1]
            else:
                json_dict[key] = value
        for i in range(len(json_dict['shapes'])):
            for j in range(len(json_dict['shapes'][i]['points'])):
                new_point = [aug_points.keypoints[k].x, aug_points.keypoints[k].y]
                json_dict['shapes'][i]['points'][j] = new_point
                k += 1
        return json_dict


""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------- Construct H and G in GF(2) [7, 4, 3] ---------------------------------------------------------- """
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------------------------------------------------------------------------------------------------------- """
def get_H_G():
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1]
    ], dtype=np.int8)
    G = np.array([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=np.int8)
    return H, G

def get_code_words(bitstreams):
    """ bitstreams.dim == 2 """
    _, G = get_H_G()
    code_words = []
    for bitstream in bitstreams:
        bitstream = bitstream.reshape(-1, 4)
        code_words.append([np.matmul(msg, G) % 2 for msg in bitstream])
    
    return np.array(code_words)

def get_err_corr_bits(bitstreams):
    """ err_corr_bits.shape = len(chars), 9. 
        Not include check bits """
    err_corr_bits = get_code_words(bitstreams)[:, :, -3:].reshape(-1, 9)
    return err_corr_bits

def check_code_word(code_words):
    H, _ = get_H_G()
    for code_word in code_words:
        result = np.matmul(H, np.transpose(code_word)) % 2
        if np.count_nonzero(result) != 0:
            print("Error!")
            return False
    
    print("G and H are both correct!")

def get_hamming_cbarcode(color_matrix, err_corr_bits):

    undetermined_bits = err_corr_bits[:, 2::3].reshape(-1).tolist()  # 无法直接append至颜色的bit
    print(f"The redundant bits is : {undetermined_bits}")
    haming_cbarcodes = []

    # append redundant bit to corresponding color bar
    for i in range(err_corr_bits.shape[0]):
        bits = [err_corr_bits[i][j] for j in range(err_corr_bits.shape[1]) 
                                    if (j + 1) % 3 != 0]
        haming_cbarcodes.append([(color_matrix[i][j], bits[j]) for j in range(len(bits))])
    
    # append undetermined bits to checkbits and STOP
    check_STOP_with_width = []
    if len(undetermined_bits) <= len(color_matrix[-1]) + len(color_matrix[-2]):
        check_STOP = color_matrix[-2] + color_matrix[-1]
        for i in range(len(check_STOP)):
            if i < len(undetermined_bits):
                check_STOP_with_width.append((check_STOP[i], undetermined_bits[i]))
            else:
                check_STOP_with_width.append((check_STOP[i], 0))

        haming_cbarcodes.append(check_STOP_with_width[:6])
        haming_cbarcodes.append(check_STOP_with_width[6:])

    return haming_cbarcodes

def split_hamming_cbarcode(hamming_cbarcodes):
    color_mats, err_corr_bits = [], []
    for haming_cbarcode in hamming_cbarcodes:
        color_mats.append([haming_bar[0] for haming_bar in haming_cbarcode])
        err_corr_bits.append([haming_bar[1] for haming_bar in haming_cbarcode])
    
    num_chars = len(color_mats[: -2])
    
    append2color_bits = np.array(err_corr_bits[:-2], dtype=np.int8).reshape(num_chars, -1, 2)
    redundant_bits = (err_corr_bits[-2] + err_corr_bits[-1])[:num_chars * 3]

    err_corr_bits, it = [], iter(redundant_bits)

    for i in range(append2color_bits.shape[0]):
        for j in range(append2color_bits.shape[1]):
            err_corr_bits.append(append2color_bits[i][j].tolist() + [next(it)])
    
    err_corr_bits = np.array(err_corr_bits, dtype=np.int8).reshape(num_chars, -1)

    return color_mats, err_corr_bits

def convert2code_words(hamming_cbarcodes):
    color_mats, err_corr_bits = split_hamming_cbarcode(hamming_cbarcodes)
    err_corr_bits = np.array(err_corr_bits, dtype=np.int8).reshape(-1, 3)
    bitstreams = np.array(color2bitstream(color_mats)[: -2], dtype=np.int8).reshape(-1, 4)
    
    code_words = [bitstreams[i].tolist() + err_corr_bits[i].tolist() 
                  for i in range(len(bitstreams))]
    return np.array(code_words, dtype=np.int8).reshape(len(hamming_cbarcodes[:-2]), -1, 7)

def get_error_syndrome(code_words):
    """ S = R · H.T = (C + E) · H.T = E · H.T """
    H, _ = get_H_G()
    S = np.matmul(code_words, H.T) % 2
    return S

def correct_errors(code_words):
    """ Try to correct errors. """
    # calculate error syndrome, S.shape = num_chars, 3, n - k
    S = get_error_syndrome(code_words)

    print(f"The syndrome matirx is:\n{S}")

    # get non-zero ids in S
    nonzero_ids = [(i, j) for i in range(S.shape[0]) for j, s in enumerate(S[i]) 
                    if s.tolist() != np.zeros_like(s).tolist()]
    
    if nonzero_ids != []:
        print("The error code word position is :")
        for nonzero_idx in nonzero_ids:
            print(f"\tThe character index is:{nonzero_idx[0]}\n"
                  f"\tThe code word index is:{nonzero_idx[1]}\n")

    H, _ = get_H_G()
    H_T = [h_T.tolist() for h_T in H.T]
    
    corr_pos_list = []
    for nonzero_idx in nonzero_ids:
        if H_T.count(S[nonzero_idx].tolist()) != 0:
            # try to correct errors but maybe fail.e.g. green(01) -> blue(10)
            # corr_pos means: index_char, index_code_words, error_pos
            corr_pos_list.append((nonzero_idx[0], nonzero_idx[1], 
                                    H_T.index(S[nonzero_idx].tolist())))
        else:
            return None
    
    # invert target error pos
    for corr_pos in corr_pos_list:
        if code_words[corr_pos] == 0:
            code_words[corr_pos] = 1
        else:
            code_words[corr_pos] = 0

    # check if correct successfully
    S_check = get_error_syndrome(code_words)
    nonzero_ids_check = [(i, j) for i in range(S_check.shape[0]) for j, s in enumerate(S_check[i]) 
                    if s.tolist() != np.zeros_like(s).tolist()]

    if nonzero_ids_check == []:
        print(f"Correct successfully by parity check matrix...")
        return code_words
    else:
        print("Failed. Too many bits get error!")
        return None


""" -------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------------------------------------------------- """
""" --------------------------------------------------Get from .txt file------------------------------------------------------------ """
""" -------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------------------------------------------------- """
def sort_cls_bbox_list(cls_bbox_list):
    """ 
    sorted bbox_list sequentially by left-top_corner -> right-bottom_corner.
    bbox_list: a list of tuple, every tuple element: (category_id, [left_x, left_y, width, height]),
    function sorts by L2 norm respective the leftest bbox.
    """
    if cls_bbox_list is not None and len(cls_bbox_list) > 1:
        sorted_index, sroted_bboxes = [], []
        left_x, left_y = 0, 0

        for _ in range(len(cls_bbox_list)):
            L2 = [(left_x - bbox[1][0]) ** 2 + (left_y - bbox[1][1]) ** 2 
                       for index, bbox in enumerate(cls_bbox_list) if index not in sorted_index]
            L2_idx = [index for index, _ in enumerate(cls_bbox_list) if index not in sorted_index]
            
            sorted_index.append(L2_idx[L2.index(min(L2))])
            sroted_bboxes.append(cls_bbox_list[sorted_index[-1]])
            left_x, left_y = sroted_bboxes[-1][1][0], sroted_bboxes[-1][1][1]
        
        return sroted_bboxes
    else:
        # print("Warning: bbox_list is bad argument!")
        return cls_bbox_list

def get_classes_bboxes(bbox_path, scroed_threshold=0.0):
    """
    bbox_path: file path of bbox.txt,

    """
    cls_bbox_list = []
    with open(bbox_path, mode='r') as bbox_file:
        lines = bbox_file.readlines()
        for line in lines:
            cls_bbox_score = line.strip('[').strip(']\n').split(', ')
            if float(cls_bbox_score[-1]) > scroed_threshold:
                cls_bbox_list.append((int(cls_bbox_score[0]), 
                                     [int(corrdinate) for corrdinate in cls_bbox_score[1:-1]]))
        cls_bbox_list = sort_cls_bbox_list(cls_bbox_list)
    
    return [cls_bbox[0] for cls_bbox in cls_bbox_list], [cls_bbox[1][:] for cls_bbox in cls_bbox_list]

def classes2color_mat(bbox_path, score_threshold=0.0):
    """  
    classes    ->    color_matrix_list
    [[category_id_1, ategory_id_2, ategory_id_3, ...] ...] -> [[[c_1, c_2, c_3, ...], [c_n, ...], ... ]]
    """

    classes, _ = get_classes_bboxes(bbox_path, score_threshold)
    
    category_id_map = {0: 'blue', 1: 'green', 2: 'red', 3: 'black'}
    stop_color = ['green','blue','blue','red','red','red','green']

    color_list = [category_id_map[cls] for cls in classes]

    if len(color_list) > 7: # respect to full barcode
            color_matrix = [color_list[index: index + 6] for index in range(0, len(color_list[:-7]), 6)]
                
    else:   # respect to short barcode
            color_matrix = [color_list]

    return color_matrix

def batch_classes2color_mat(dir_path, score_threshold=0.0):
     """ 从指定文件夹下批量读入并转换 """
     txt_path_list = glob.glob(os.path.join(dir_path, '*.txt'))
     color_mat_list = [classes2color_mat(p, score_threshold) for p in txt_path_list]
     return txt_path_list, color_mat_list
       
       
""" ----------------------------------------------------------------------------------------------------------------------- """          
""" ----------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------Get Color Barcode from 'image_id' image---------------------------------------- """
""" ----------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------------------------------------------------------------------------- """
def sort_bbox_list(bbox_list):
    """
    sorted bbox_list sequentially by left-top_corner -> right-bottom_corner.
    bbox_list: a list of tuple, every tuple element: (category_id, [left_x, left_y, width, height]),
    function sorts by L2 norm respective the leftest bbox.
    """
    if bbox_list is not None and len(bbox_list) > 1:
        sorted_index, sroted_bboxes = [], []
        left_x, left_y = 0, 0

        for _ in range(len(bbox_list)):
            L2 = [(left_x - bbox[1][0]) ** 2 + (left_y - bbox[1][1]) ** 2
                  for index, bbox in enumerate(bbox_list) if index not in sorted_index]
            L2_idx = [index for index, _ in enumerate(bbox_list) if index not in sorted_index]

            sorted_index.append(L2_idx[L2.index(min(L2))])
            sroted_bboxes.append(bbox_list[sorted_index[-1]])
            left_x, left_y = sroted_bboxes[-1][1][0], sroted_bboxes[-1][1][1]

        return sroted_bboxes
    else:
        # print("Warning: bbox_list is bad argument!")
        return bbox_list


def get_image_id_bbox(json_utils, dets_json_path, score_threshold=0.0, image_id=0):
    """ 从detection json中提取指定image_id的bounding box 坐标  """
    dets_dicts = json_utils.read_json_files(dets_json_path)

    bbox_list = [(dets_dict["category_id"], dets_dict["bbox"])
                 for dets_dict in dets_dicts if dets_dict["score"] > score_threshold
                 and dets_dict["image_id"] == image_id]

    sorted_bboxes = sort_bbox_list(bbox_list)
    return sorted_bboxes


def get_images_bbox(dets_json_path, image_path, json_utils=JsonUtils(), score_threshold=0.0):
    """
    image_path 下的图片数量应与json文件中的image_id相对应

    Return:
        image_id_bboxes: a list of bbox_list,

            [ [ [(category_id_1, [x1, y1, w1, h1]), (category_id_2, [x2, y2, w2, h2]),...], ... ], ... ]

            image_id_bboxes[i][j][k][l]: image_id=i, j bbox, k=(0, 1) category or bbox, l=(0, 1, 2, 3) x/y/w/h
    """
    image_id_bboxes = [[] for _ in range(len(os.listdir(image_path)))]

    for image_id in range(len(image_id_bboxes)):
        sorted_bboxes = get_image_id_bbox(json_utils, dets_json_path, score_threshold, image_id)
        image_id_bboxes[image_id].append(sorted_bboxes)

    print(f"'image_id' counts: {len(image_id_bboxes)}")
    return image_id_bboxes


def get_class_bbox(dets_json_path, image_path, json_utils=JsonUtils(), score_threshold=0.0):
    """
    Return:
        classes: [[category_id_1, category_id_2, category_id_3, ...]]
        bboxes: [[[x1, y1, w1, h1], [x2, y2, w2, h2], ...]...]
    """
    image_id_bboxes = get_images_bbox(dets_json_path, image_path, json_utils, score_threshold)

    classes, bboxes = [], []
    for image_id_list in image_id_bboxes:
        for sorted_boxes in image_id_list:
            classes.append([bbox[0] for bbox in sorted_boxes])
            bboxes.append([bbox[1] for bbox in sorted_boxes])
    print(classes[-1])
    return classes, bboxes


def count_empty(lists_mats, isList=True):
    """ 检测color_lists或color_matrix_lists中的空列表 """
    if isList is True:
        empty_idx = [index for index, color_list in enumerate(lists_mats)
                     if color_list == []]
    else:
        empty_idx = [index for index, color_matrix in enumerate(lists_mats)
                     if color_matrix == [[]]]
    return empty_idx


def classes2color_mat(dets_json_path, image_path, json_utils=JsonUtils(), score_threshold=0.0):
    """
    classes    ->    color_matrix_list
    [[category_id_1, ategory_id_2, ategory_id_3, ...] ...] -> [[[c_1, c_2, c_3, ...], [c_n, ...], ... ]]
    """

    classes, _ = get_class_bbox(dets_json_path, image_path, json_utils, score_threshold)

    category_id_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'black'}
    stop_color = ['green', 'blue', 'blue', 'red', 'red', 'red', 'green']

    color_lists, color_matrix_list = [], []
    for image_id_cls in classes:
        color_lists.append([category_id_map[cat_id] for cat_id in image_id_cls])

    empty_idx = count_empty(color_lists)
    print(f"Empty image_id counts: {len(empty_idx)}.\n",
          f"Empty 'image_id' is: {empty_idx}")

    for image_id, color_list in enumerate(color_lists):

        if image_id not in empty_idx:

            if len(color_list) > 7:  # respect to full barcode
                color_matrix = [color_list[index: index + 6] for index in range(0, len(color_list[:-7]), 6)]

            else:  # respect to short barcode
                color_matrix = [color_list]

            color_matrix_list.append(color_matrix)

            # to be easy for decoding process
            #       color_matrix_list[image_id].append(stop_color)

        else:
            color_matrix_list.append([[]])

    return color_matrix_list


""" Draw Color Barcode """

def draw_color_barcode(color_matrix, height=720, width=1024, channels=3, delay=0):
    """绘制color_barcode"""

    def color2scalar(color_matrix):
        color2bit_map = {'red': (0, 0, 255),
                         'green': (0, 255, 0),
                         'blue': (255, 0, 0),
                         'black': (0, 0, 0)}
        stop_color = ['green','blue','blue','red','red','red','green']
        
        colors_m = color_matrix[:]
        return [color2bit_map[color] for colors in (colors_m) for color in colors]
    
    def draw_bar(start_pos, num_bars):
        bar_pixel = (width - start_pos * 2) // ((num_bars) * 2)
        pos = [(start_pos + 2 * i * bar_pixel) for i in range(num_bars)]
        #pos = [(start_pos + i * (bar_pixel + 5)) for i in range(num_bars)]
        return pos, bar_pixel

    # color_matrix输入形状：num_chars, 6
    color_list = color2scalar(color_matrix)
    num_bars = len(color_list)
    print(f"number of bars is : {num_bars}")
    whiteboard = np.full((height, width, channels), 255, dtype=np.uint8)
    start_pos = int(0.1 * whiteboard.shape[1])
    pos, bar_pixel = draw_bar(start_pos, num_bars)
    for i, p in enumerate(pos):
        cv2.rectangle(whiteboard, pt1=(p, int(0.1 * height)), 
                      pt2=(p + bar_pixel, int(0.9 * height)), color=color_list[i], thickness=-1)
    cv2.imshow("Color Barcode", whiteboard)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def show_cbarcode_images(color_matrix_lists, height=720, width=1024, channels=3):
    empty_index = count_empty(color_matrix_lists, isList=False)
    for image_id, color_matrix in enumerate(color_matrix_lists):
        if image_id not in empty_index:
            draw_color_barcode(color_matrix, height, width, channels, delay=0)

def draw_hamming_cbarcodes(hamming_cbarcodes, height=720, width=1024, channels=3, delay=0):
    """绘制color_barcode"""

    def color2scalar(color_matrix):
        color2bit_map = {'red': (0, 0, 255),
                         'green': (0, 255, 0),
                         'blue': (255, 0, 0),
                         'black': (0, 0, 0)}
        stop_color = ['green','blue','blue','red','red','red','green']
        
        colors_m = color_matrix[:]
        return [color2bit_map[color] for colors in (colors_m) for color in colors]
    
    def draw_bar(start_pos, num_bars, err_corr_bits):
        w = (width - start_pos * 2) // ((num_bars) * 2 * 2)
        bar_pixels = [w if bit == 0 else w * 2 for bit in err_corr_bits]
        
        pos = []
        for i, bar_pixel in enumerate(bar_pixels):
            if i == 0:
                pos = [start_pos]
            else:
                pos.append(pos[-1] + bar_pixels[i - 1] + w)
        print(len(pos))
        #pos = [(start_pos + i * (bar_pixel + 5)) for i in range(num_bars)]
        return pos, bar_pixels

    # color_matrix输入形状：num_chars, 6
    color_matrix, err_corr_bits = split_hamming_cbarcode(hamming_cbarcodes)
    color_list = color2scalar(color_matrix)
    err_corr_bits = err_corr_bits.reshape(-1).tolist()
    err_corr_bits += [0] * (len(color_list) - len(err_corr_bits))
    num_bars = len(color_list)
    print(f"number of bars is : {num_bars}")
  
    whiteboard = np.full((height, width, channels), 255, dtype=np.uint8)
    start_pos = int(0.1 * whiteboard.shape[1])
    pos, bar_pixel = draw_bar(start_pos, num_bars, err_corr_bits)
    for i, p in enumerate(pos):
        cv2.rectangle(whiteboard, pt1=(p, int(0.1 * height)), 
                      pt2=(p + bar_pixel[i], int(0.9 * height)), color=color_list[i], thickness=-1)
    cv2.imshow("Color Barcode", whiteboard)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


""" ----------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------- Encoding&Decoding -------------------------------------------------- """
""" ----------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------------------------------------------------------------------------- """
def encoding_from_in_str(in_string, code_type='A', c128AB=get_c128AB()):
    """ Convert input string to barcode/cbarcode """
    barcode = chars2barcode(append_checkbit(in_string, c128AB), c128AB, code_type)

    color_barcode = bitstream2color(values2bits(barcode))
    color_stop = bitstream2color(values2bits([c128_stop]))

    barcode.append(c128_stop)   # append stop bits
    color_barcode.append(color_stop[0])  #append stop color bits

    print(f"Input string with check bits is {in_string}", "\noutput color barcode is :\n", color_barcode)

    return barcode, color_barcode


def decoding_from_color_matrix(color_matrix, code_type='B', is_stop_exist=False, c128AB=get_c128AB()):
    """ Convert input string to barcode/cbarcode """
    bitstreams = color2bitstream(color_matrix)

    if is_stop_exist:
        bitstreams = bitstreams[:-1]

    out_string = barcode2chars(bits2values(bitstreams), c128AB, code_type)
    barcode = bits2values(bitstreams).tolist()

    color_stop_bits = bitstream2color(values2bits([c128_stop]))

    if is_stop_exist:
        if color_stop_bits[0] != color_matrix[-1]:
            print("Error: Stop bits is bad!")
            return -1

    out_string.append('STOP')
    barcode.append(c128_stop)

    print(f"Color barcode is \n{color_matrix}", "\n\nOutput color barcode is :\n", out_string)

    return barcode, out_string


def decoding_from_hamming_cbarcode(hamming_cbarcodes, code_type='B', is_stop_exist=False, c128AB=get_c128AB()):
    color_mats, _ = split_hamming_cbarcode(hamming_cbarcodes)
    barcode, _ = decoding_from_color_matrix(color_mats, code_type, is_stop_exist, c128AB)
    code_words = convert2code_words(hamming_cbarcodes)

    if barcode is None:
        """ Try to correct errors. """
        code_words = correct_errors(code_words)
        
        if code_words is not None:
            code_words = code_words[:, :, : 4].reshape(code_words.shape[0], -1)
            
            ret_string = barcode2chars(bits2values(code_words), c128AB)
            #print(ret_string)
            if len(ret_string) == code_words.shape[0]:
                print(f"Decodes from color barcode result is :\n{ret_string}")
                return ret_string
            print("Failed! Get too much error.")
            return None
        
        else:
            return None










""" ------------------------------------------ Example ------------------------------------------"""
c128AB = get_c128AB()

def example_1():
    """ conversion among characters, bitstream, barcode """
    
    in_string = ['START_B', '7', '7', '1']
    barcode = chars2barcode(in_string, c128AB, 'A')
    
    print(f"Input string is :\n{in_string}", "\nOutput barcode is :\n", barcode)
    
    bitstreams = values2bits(barcode)
    
    print(f"\nResult of converting to bitstream is :\n{bitstreams}")
    print(f"\nResult of converting to characters is: \\
          \n{barcode2chars(bits2values(bitstreams), c128AB, get_c128_type(barcode))}")
        

def main():
    # test
    in_string = ['START_B', '5', '2', '0']

    _, color_barcode = encoding_from_in_str(in_string, 'B')

    dets_json_path = "D:/Train/models/yolact-master/results/bbox_detections.json"
    image_path = "D:/Train/models/yolact-master/data/val"
    c2mats = classes2color_mat(dets_json_path, image_path, json_utils=JsonUtils(), score_threshold=0.15)
    _, out_string = decoding_from_color_matrix(c2mats[10], is_stop_exist=False)


if __name__ == '__main__':
    main()