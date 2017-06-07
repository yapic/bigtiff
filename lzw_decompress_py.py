# Python version of LZW decompressor
# much slower than the cython version!

import numpy as np

class EoiCode(object):
    pass

class ClearCode(object):
    pass

TIFF_TABLE = [
    bytes([i]) for i in range(256)
]
TIFF_TABLE.append(ClearCode)
TIFF_TABLE.append(EoiCode)

TIFF_CODE_LEN = [9, 12]


class BitReader(object):
    def __init__(self, buf):
        self.buf = buf
        self.curr = 0
        self.read = 0

    mask = [0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f]

    def readbits(self, n):
        left = n - self.read
        if left > 8:
            self.curr = (self.curr << 8) | (next(self.buf) & 0xff)
            left -= 8
        self.read = 8 - left
        next_byte = next(self.buf) & 0xff

        print(next_byte, self.read)
        print('a', (self.curr << left), '', (next_byte >> self.read))
        r = (self.curr << left) | (next_byte >> self.read)
        self.curr = next_byte & self.mask[self.read]
        return r


class LZWError(Exception):
    pass


def lzw_decompress(init_table, code_len, buf):
    min_code_len, max_code_len = code_len
    code_len = min_code_len
    table = []+init_table
    reader = BitReader(iter(buf))
    out = b''

    while True:
        code = reader.readbits(code_len)
        if code < len(table):
            value = table[code]
        else:
            value = None

        if value == EoiCode:
            break
        elif value == ClearCode:
            print('CLEAR')
            table = [] + init_table
            code_len = min_code_len

            code = reader.readbits(code_len)
            value = table[code]

            if value == EoiCode:
                break

            out += value
        else:
            if value is not None:
                print('1', old_code, code, code_len)
                out += value
                table.append(table[old_code] + value[:1])
            else:
                print('2', old_code, code, code_len)
                old_value = table[old_code]
                string = old_value + old_value[:1]
                out += string
                table.append(string)
        old_code = code

        if len(table) == 511:
            code_len = 10
        elif len(table) == 1023:
            code_len = 11
        elif len(table) == 2047:
            code_len = 12

    return out


def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')


if __name__ == '__main__':
    b = [256, 7, 258, 8, 8, 258, 6, 257]
    b = ('{:09b}'*len(b)).format(*b)
    b = bitstring_to_bytes(b)
    res = lzw_decompress(TIFF_TABLE, TIFF_CODE_LEN, b)

    np.testing.assert_array_equal(np.frombuffer(res, dtype='uint8'), [7,7,7,8,8,7,7,6])

