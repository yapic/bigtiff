import sys
import itertools

import numpy as np

import tif_format
from strip import Strip

class Image2dIterator(object):
    def __init__(self, first_ifd):
        self.next_ifd = first_ifd


    def __next__(self):
        if self.next_ifd is None:
            raise StopIteration

        ifd = self.next_ifd
        self.next_ifd = self.next_ifd.next_ifd
        return Image2d(ifd)


def if_none(value, optional):
    if value is None:
        return optional
    else:
        return value

class Image2d(object):
    '''
    This represents one image in the TIF file.

    WARNING:
    Although this class is called Image2d it can either be a grayscale image
    dim=(X,Y,1) or a RGB image dim=(X,Y,1).
    This is due to the way TIFF saves the images in the file.
    '''
    def __init__(self, ifd):
        self.ifd = ifd

    @property
    def tags(self):
        tags = {}
        for i, entry in enumerate(self.ifd.entries):
            if entry.is_value:
                tags[entry.tag.name] = entry.values.array
            else:
                tags[entry.tag.name] = entry.external_values.array

        return tags

    @property
    def dtype(self):
        tags = self.tags

        endian = self.ifd._root.endian
        endian = {
            endian.be: '>',
            endian.le: '>',
        }[endian]

        typ = tags['sample_format'][0]
        typ = {
             1: 'u',
             2: 'i',
             3: 'f',
        }[typ]

        length = np.ceil(tags['bits_per_sample'][0] / 8.0)
        return '{}{}{}'.format(endian, typ, int(length))

    @property
    def strips(self):
        tags = self.tags
        strip_offsets = tags.get('strip_offsets', [])
        strip_byte_counts = tags.get('strip_byte_counts', [])
        compression = tags.get('compression', [1])*len(strip_offsets)

        if strip_offsets is None or strip_byte_counts is None:
            return []

        return [Strip(self.ifd._io, offset, length, compr)
                for offset, length, compr in zip(strip_offsets, strip_byte_counts, compression)]


    def __getitem__(self, slices):
        assert len(slices) == 3

        tags = self.tags

        strip_offsets = tags.get('strip_offsets', [])
        rows_per_strip = tags.get('rows_per_strip', [])

        if len(rows_per_strip) == len(strip_offsets) - 1:
            # last strip is missing
            rows_per_strip.append(sys.maxsize)

        if len(rows_per_strip) == 1:
            # we have only one value for all strips
            rows_per_strip = rows_per_strip * len(strip_offsets)

        start_row_per_strip = np.hstack([[0], np.cumsum(rows_per_strip[:-1])])
        end_row_per_strip = start_row_per_strip + rows_per_strip

        first = None
        strip_list = []
        for strip, strip_start, strip_stop in zip(self.strips, start_row_per_strip, end_row_per_strip):
            if strip_start <= (slices[0].start or 0):
                if strip_start <= (slices[0].stop or tags['image_length']):
                    first = first or strip_start
                    strip_list.append(strip.read(self.dtype))
                else:
                    break

        channels = tags.get('samples_per_pixel', [1])[0]
        if len(strip_list) == 0:
            return np.array([]).reshape([0,0, channels])

        data = np.concatenate(strip_list)
        data = data.reshape([-1, tags.get('image_width', [0])[0], channels])

        slices = list(slices)
        slices[0] = slice(slices[0].start - first, slices[0].stop - first, slices[0].step)
        return data[slices[0], slices[1], slices[2]]

