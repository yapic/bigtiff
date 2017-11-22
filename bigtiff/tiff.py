import sys
import io as _io

import numpy as np

import tif_format
from kaitaistruct import KaitaiStream

from image2d import Image2dIterator
from writer import Writer
import representation

class Tiff(tif_format.TifFormat):
    def __iter__(self):
        if not self.header.is_big_tiff:
            return Image2dIterator(self.header.first_ifd)
        else:
            return Image2dIterator(self.header.first_ifd_big_tiff)


    @classmethod
    def from_file(cls, filename, closefd=True):
        f = open(filename, 'br+') # we need write permission for memmap
        try:
            return cls(KaitaiStream(f))
        except Exception:
            if closefd:
                f.close()
            raise

        if closefd:
            f.close()


    @classmethod
    def write(cls, images, io=None, big_tiff=True, closefd=True):
        '''
        Writes all `images` (which should be a list of 3D (H,W,C) numpy arrays).
        Returns the io object

        If `io` is a string, we write the image to that path,
        if `io` is None, we return a io.BytesIO (io.getbuffer() returns a buffer)
        '''
        if io is None:
            io = Writer(_io.BytesIO())
        elif type(io) is str:
            io = Writer(open(io, 'wb'))
        else:
            io = Writer(io)

        if sys.byteorder == 'little':
            io.write_bytes(b'II')
        else:
            io.write_bytes(b'MM')

        if big_tiff:
            offset_width = 8
            write_offset = io.write_u64
            io.write_u16(43) # magic
            io.write_u16(8) # offset width
            io.write_u16(0)
        else:
            offset_width = 4
            write_offset = io.write_u32
            io.write_u16(42) # magic

        next_ifd_position = len(io) + 8
        write_offset(next_ifd_position)

        for i, img in enumerate(images):
            io_data = Writer(_io.BytesIO())
            cls._write_image(io, io_data, img, len(io), big_tiff)

            if i == len(images) - 1:
                next_ifd_position = 0
            else:
                next_ifd_position = len(io) + offset_width + len(io_data) + img.nbytes

            write_offset(next_ifd_position)
            io.write_bytes(io_data.get_bytes())

            if type(img) is PlaceHolder:
                io.write_hole(img.nbytes)
            else:
                io.write_bytes(img.tobytes())

        if closefd:
            io.close()

        return io.io


    @classmethod
    def _write_image(cls, io_ifd, io_data, img, total_offset, big_tiff):
        Tag = tif_format.TifFormat.Tag
        TagType = tif_format.TifFormat.TagType

        if big_tiff:
            offset_width = 8
            offset_type = TagType.u8
            write_offset = io_ifd.write_u64
        else:
            offset_width = 4
            offset_type = TagType.u4
            write_offset = io_ifd.write_u32

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
                (Tag.bits_per_sample, TagType.u2, [img.dtype.itemsize * 8]),
                (Tag.sample_format, TagType.u2, [kind]),
                (Tag.x_resolution, TagType.u_ratio, [(1, 1)]),
                (Tag.y_resolution, TagType.u_ratio, [(1, 1)]),
                (Tag.strip_byte_counts, TagType.u4, [img.nbytes]),
                (Tag.rows_per_strip, TagType.u4, [height]),
                ]
        tags = sorted(tags, key=lambda t: t[0].value)

        # the len(tags) + 1 is for strip_offsets which we write later
        ifd_len = offset_width + (len(tags) + 1) * (20 if big_tiff else 12) + offset_width
        write_offset(len(tags) + 1)

        for t in tags:
            offset = total_offset + ifd_len + len(io_data)
            io_now = Writer(_io.BytesIO())
            io_later = Writer(_io.BytesIO())
            cls._write_tag(io_now, io_later, t, offset, big_tiff)
            io_ifd.write_bytes(io_now.get_bytes())
            io_data.write_bytes(io_later.get_bytes())

        strip_offsets = [ total_offset + ifd_len + len(io_data)]
        t = (Tag.strip_offsets, offset_type, strip_offsets)
        offset = total_offset + ifd_len + len(io_data)

        cls._write_tag(io_ifd, io_data, t, offset, big_tiff)

        return io_ifd, io_data


    @classmethod
    def _write_tag(cls, io_now, io_later, tag, offset, big_tiff):
        TagType = tif_format.TifFormat.TagType
        if big_tiff:
            offset_width = 8
            offset_type = TagType.u8
            write_offset = io_now.write_u64
        else:
            offset_width = 4
            offset_type = TagType.u4
            write_offset = io_now.write_u32

        kind, typ, values = tag
        old_now_len = len(io_now)

        io_now.write_u16(kind.value)
        io_now.write_u16(typ.value)
        write_offset(len(values))

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

        io = Writer(_io.BytesIO())
        for v in values:
            io.write_tag_value(typ, v, offset)

        if inline:
            io_now.write_bytes(io.get_bytes())
            io_now.write_bytes(b'\x00'*(4 + 2*offset_width - len(io_now)))
        else:
            write_offset(offset)
            io_later.write_bytes(io.get_bytes())

        if big_tiff:
            assert old_now_len == len(io_now) - 20
        else:
            assert old_now_len == len(io_now) - 12

        return io_now, io_later


class PlaceHolder(object):
    '''
    A place holder that can be used if you do not really want to
    write any image data (behaves like a numpy array).
    '''
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def nbytes(self):
        return self.dtype.itemsize * self.size


if __name__ == '__main__':
    main()

