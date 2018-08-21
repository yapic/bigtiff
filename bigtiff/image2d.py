import re
import sys
import itertools
from collections import OrderedDict

import numpy as np

import bigtiff.tif_format
from bigtiff.strip import Strip
from bigtiff.memmap import memmap as my_memmap

class Image2dIterator(object):
    def __init__(self, first_ifd):
        self.next_ifd = first_ifd


    def __next__(self):
        if self.next_ifd is None:
            raise StopIteration

        ifd = self.next_ifd
        self.next_ifd = self.next_ifd.next_ifd
        return Image2d(ifd)


class Image2d(object):
    '''
    This represents one image in the TIF file.

    WARNING:
    Although this class is called Image2d it can either be a grayscale image
    dim=(X,Y,1) or a RGB image dim=(X,Y,3).
    This is due to the way TIFF saves the images in the file.
    '''
    def __init__(self, ifd):
        self.ifd = ifd

        tags = self.tags
        self.height = tags['image_length'][0]
        self.width = tags['image_width'][0]

        # self.write_image = np.ma.masked_all([height, width, self.n_channels])


    @property
    def n_channels(self):
        channels = self.tags['samples_per_pixel'][0]
        return channels


    @property
    def axes(self):
        # ImageJ storage order is TZCXY
        desc = self.tags['image_description'][0].string

        images = re.search('images=([0-9]+)', desc)
        channels = re.search('channels=([0-9]+)', desc)
        slices = re.search('slices=([0-9]+)', desc)
        frames = re.search('frames=([0-9]+)', desc)

        axes = [('X', 999, self.width), ('Y', 888, self.height)]
        if channels:
            axes.append(('C', channels.start(), channels.group(1)))
        else:
            axes.append(('C', -1, self.n_channels))
        if slices:
            axes.append(('Z', slices.start(), slices.group(1)))
        if frames:
            axes.append(('T', frames.start(), frames.group(1)))

        axes = OrderedDict((a, int(value)) for a, start, value in sorted(axes, key=lambda pair: pair[1]))
        return axes


    @property
    def shape(self):
        return (self.width, self.height, self.n_channels)


    @property
    def tags(self):
        tags = {
            'samples_per_pixel': [1],
            'sample_format': [1],
            'strip_offsets': [],
            'strip_byte_counts': [],
            'compression': [1],
            'image_width': [0],
            'rows_per_strip': [],
            'image_length': [0],
        }

        for entry in self.ifd.entries:
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
            endian.le: '<',
        }[endian]

        typ = tags['sample_format'][0]
        typ = {
             1: 'u',
             2: 'i',
             3: 'f',
        }[typ]

        length = np.ceil(tags['bits_per_sample'][0] / 8.0)
        return np.dtype('{}{}{}'.format(endian, typ, int(length)))


    @property
    def strips(self):
        '''List of Strips (data slices) for this image'''
        tags = self.tags
        strip_offsets = tags['strip_offsets']
        strip_byte_counts = tags['strip_byte_counts']

        if strip_offsets is None or strip_byte_counts is None:
            return []

        compression = tags['compression'] * len(strip_offsets)

        return [ Strip(self.ifd._io, offset, length, compr)
                 for offset, length, compr in zip(strip_offsets, strip_byte_counts, compression) ]


    def __getitem__(self, slices):
        '''
        Deprecated: Use self.memmap() instead.
        Get image data (use three indices: H, W, C).
        '''
        assert len(slices) == 3

        first_row = None
        strip_list = []
        for i, strip, strip_start, strip_stop in strip_iterator((slices[0].start or 0),
                                                                (slices[0].stop or tags['image_length'])):
            first_row = first_row or strip_start
            strip_list.append(strip.read(self.dtype_tif))

        data = np.concatenate(strip_list)
        data = data.reshape([-1, tags['image_width'][0], self.n_channels])

        slices = list(slices)
        slices[0] = slice(slices[0].start - first_row, slices[0].stop - first_row, slices[0].step)
        return data[slices[0], slices[1], slices[2]]


    def strip_iterator(self, start_row, stop_row):
        '''
        Iterate over strips that include rows from start_row to stop_row
        '''
        tags = self.tags
        strip_offsets = tags['strip_offsets']
        rows_per_strip = tags['rows_per_strip']

        if len(rows_per_strip) == len(strip_offsets) - 1:
            # last strip is missing
            rows_per_strip.append(sys.maxsize)

        if len(rows_per_strip) == 1:
            # we have only one value for all strips
            rows_per_strip = rows_per_strip * len(strip_offsets)

        start_row_per_strip = np.hstack([[0], np.cumsum(rows_per_strip[:-1])])
        end_row_per_strip = start_row_per_strip + rows_per_strip

        for i, strip, strip_start, strip_stop in enumerate(zip(self.strips, start_row_per_strip, end_row_per_strip)):
            if strip_start <= start_row:
                if strip_start <= stop_row:
                    yield i, strip, strip_start, strip_stop
                else:
                    break


    def __setitem__(self, slices, values):
        '''Deprecated: Use self.memmap() instead'''
        raise NotImplemented


    def flush(self):
        '''
        Deprecated: Use self.memmap() instead.
        Write assignments done using __setitem__() to disk.
        '''
        raise NotImplemented

        tags = self.tags
        width = tags['image_length']

        edges = np.ma.notmasked_edges(self.write_image, axis=0)
        if edges is None:
            return

        for row_start, row_stop in edges:
            first_row = None
            for i, strip, strip_start, strip_end in strip_iterator(row_start, row_stop):
                assert tags['compression'][i] == 1
                first_row = first_row or stip_start


    def memmap(self):
        '''Returns an `numpy.memmap()`-ed array of the image'''
        tags = self.tags
        H = tags['image_length'][0]
        W = tags['image_width'][0]
        C = tags['samples_per_pixel'][0]

        strips = self.strips
        pos = strips[0].offset
        for s in strips:
            if pos != s.offset:
                msg = 'Cannot memmap non-consecutive image strips (image has {} strips)'
                raise NotImplementedError(msg.format(len(strips)))

            pos += s.length

        if strips[0].compression != 1:
            raise NotImplementedError('Cannot memmap compressed images')

        array = my_memmap(strips[0].io._io, mode='r+', dtype=self.dtype,
                          shape=(H, W, C), offset=strips[0].offset)

        if array.shape[-1] == 1:
            return array[:, :, 0]
        else:
            return array
