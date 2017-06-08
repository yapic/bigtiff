import sys

import tif_format

from image2d import Image2dIterator
from writer import Writer
import representation

class Tiff(tif_format.TifFormat):
    def __iter__(self):
        if self.header.is_big_tiff is None:
            return Image2dIterator(self.header.first_ifd)
        else:
            return Image2dIterator(self.header.first_ifd_big_tiff)

    @classmethod
    def write(cls, images, big_tiff=True):
        '''A simple image writer'''
        io = Writer()
        buf = b''

        if sys.byteorder == 'little':
            buf += io.write_u16(0x4949) # 'II'
        else:
            buf += io.write_u16(0x4d4d) # 'MM'

        if big_tiff:
            offset_width = 8
            write_offset = io.write_u64
            buf += io.write_u16(43)
            buf += io.write_u16(8) # offset width
            buf += io.write_u16(0)
        else:
            offset_width = 4
            write_offset = io.write_u32
            buf += io.write_u16(42)

        next_ifd_position = len(buf) + 8
        buf += write_offset(next_ifd_position)

        for i, img in enumerate(images):
            buf_ifd, buf_data = cls._write_image(io, img, len(buf), big_tiff)
            buf += buf_ifd

            if i == len(images) - 1:
                next_ifd_position = 0
            else:
                next_ifd_position = len(buf) + offset_width + len(buf_data) + img.nbytes

            buf += write_offset(next_ifd_position)
            buf += buf_data + img.tobytes()

        return buf


    @classmethod
    def _write_image(cls, io, img, total_offset, big_tiff):
        Tag = tif_format.TifFormat.Tag
        TagType = tif_format.TifFormat.TagType

        if big_tiff:
            offset_width = 8
            offset_type = TagType.u8
            write_offset = io.write_u64
        else:
            offset_width = 4
            offset_type = TagType.u4
            write_offset = io.write_u32

        height, width, channels = img.shape
        kind = {
            'u': 1,
            'i': 2,
            'f': 3,
        }[img.dtype.kind]

        # TODO: tags must be sorted
        tags = [
                (Tag.image_length, TagType.u4, [height]),
                (Tag.image_width, TagType.u4, [width]),
                (Tag.photometric_interpretation, TagType.u2, [1]),
                (Tag.compression, TagType.u2, [1]),
                (Tag.bits_per_sample, TagType.u2, [img.dtype.itemsize]),
                (Tag.sample_format, TagType.u2, [kind]),
                (Tag.x_resolution, TagType.u_ratio, [(1, 1)]),
                (Tag.y_resolution, TagType.u_ratio, [(1, 1)]),
                (Tag.strip_byte_counts, TagType.u4, [img.nbytes]),
                (Tag.rows_per_strip, TagType.u4, [height]),
                ]
        ifd_len = offset_width + (len(tags) + 1) * (20 if big_tiff else 12) + offset_width # the +1 is for strip_offsets which we write later

        buf_ifd = write_offset(len(tags) + 1)
        buf_data = b''
        for t in tags:
            offset = total_offset + ifd_len + len(buf_data)
            buf = cls._write_tag(io, t, offset, big_tiff)
            buf_ifd += buf[0]
            buf_data += buf[1]

        strip_offsets = [ total_offset + ifd_len + len(buf_data)]
        t = (Tag.strip_offsets, offset_type, strip_offsets)
        offset = total_offset + ifd_len + len(buf_data)
        buf = cls._write_tag(io, t, offset, big_tiff)
        buf_ifd += buf[0]
        buf_data += buf[1]

        return buf_ifd, buf_data


    @classmethod
    def _write_tag(cls, io, tag, offset, big_tiff):
        TagType = tif_format.TifFormat.TagType
        if big_tiff:
            offset_width = 8
            offset_type = TagType.u8
            write_offset = io.write_u64
        else:
            offset_width = 4
            offset_type = TagType.u4
            write_offset = io.write_u32

        kind, typ, values = tag
        buf_now = b''
        buf_later = b''

        buf_now += io.write_u16(kind.value)
        buf_now += io.write_u16(typ.value)
        buf_now += write_offset(len(values))

        itemsize = {
            TagType.u2: 2,
            TagType.u4: 4,
            TagType.u8: 8,
            TagType.u_ratio: 8,
            TagType.s2: 2,
            TagType.s4: 4,
            TagType.s8: 8,
            TagType.s_ratio: 8,
        }[typ]

        inline = itemsize * len(values) <= offset_width

        buf = b''
        for v in values:
            buf += io.write_tag_value(typ, v, offset)

        if inline:
            buf_now += buf
            buf_now += b'\x00'*(4 + 2*offset_width - len(buf_now))
        else:
            buf_now += write_offset(offset)
            buf_later += buf

        if big_tiff:
            assert len(buf_now) == 20
        else:
            assert len(buf_now) == 12

        return buf_now, buf_later


def main():
    import sys
    import numpy as np
    fname = sys.argv[1]

    tif = Tiff.from_file(fname)
    for img in tif:
        print(img.tags)
        print(img[65:100,5:101,:].shape)

    actual = Tiff.write([np.atleast_3d([1]).astype('u8'), np.atleast_3d([255]).astype('u8')])
    with open('/tmp/foo.tif', 'bw') as f:
        f.write(actual)

    expected = b'II+\x00\x08\x00\x00\x00' # header
    expected += b'\x10\x00\x00\x00\x00\x00\x00\x00' # first ifd position
    expected += b'\x0b\x00\x00\x00\x00\x00\x00\x00' # 13 entries
    expected += b'\x01\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x16\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x01\x10\x00\x01\x00\x00\x00\x00\x00\x00\x00\xfc\x00\x00\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x01\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x16\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x01\x10\x00\x01\x00\x00\x00\x00\x00\x00\x00\xf0\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff'

    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    main()

