import sys
import struct

import tif_format

class Writer(object):
    def __init__(self, endian=None):
        endian = endian or sys.byteorder

        if endian in '<>!=@':
            self.endian = endian
        else:
            self.endian = '<' if endian.startswith('l') else '>'

    def write_u16(self, v):
        return self.pack('?H', v)

    def write_i16(self, v):
        return self.pack('?h', v)

    def write_u32(self, v):
        return self.pack('?I', v)

    def write_i32(self, v):
        return self.pack('?i', v)

    def write_u64(self, v):
        return self.pack('?Q', v)

    def write_i64(self, v):
        return self.pack('?q', v)

    def write_f32(self, v):
        return self.pack('?f', v)

    def write_f64(self, v):
        return self.pack('?d', v)

    def pack(self, fmt, *v):
        return struct.pack(fmt.replace('?', self.endian), *v)

    def write_tag_value(self, typ, v, offset):
        TagType = tif_format.TifFormat.TagType

        fmt = {
            TagType.u2:      '?H',
            TagType.u4:      '?I',
            TagType.u8:      '?Q',
            TagType.u_ratio: '?II',
            TagType.s2:      '?h',
            TagType.s4:      '?i',
            TagType.s8:      '?q',
            TagType.s_ratio: '?ii',
        }[typ]

        if typ not in (TagType.u_ratio, TagType.s_ratio):
            return self.pack(fmt, v)
        else:
            return self.pack(fmt, *v)

