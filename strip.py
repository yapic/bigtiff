import numpy as np

import lzw_decompress

class Strip(object):
    def __init__(self, io, offset, length, compression):
        self.io = io
        self.offset = offset
        self.length = length
        self.compression = compression


    def read(self, dtype):
        old_position = self.io.pos()
        self.io.seek(self.offset)
        buf = self.io.read_bytes(self.length)

        if self.compression == 1:
            pass
        elif self.compression == 5:
            buf = lzw_decompress.tiff_lzw_decompress(buf)

        self.io.seek(old_position)
        return np.frombuffer(buf, dtype=dtype)


    def __repr__(self):
        msg = '<Strip offset=0x{:x} len={} compression={}>'
        return msg.format(self.offset, self.length, self.compression)

