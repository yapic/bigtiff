import time
import unittest
import os
import numpy as np

from bigtiff import Tiff, PlaceHolder

FILENAME = os.path.join(os.path.dirname(__file__), 'foo.tif')

class TestTiff(unittest.TestCase):

    @unittest.skip("FIXME test not working")
    def test_memmap(self):
        with Tiff.from_file(FILENAME) as tif:
            for img in tif:
                arr = img.memmap()
                arr[:10, :10, 0] = 99


    def test_load(self):
        with Tiff.from_file(FILENAME) as tif:
            for img in tif:
                t = img.tags

    #@unittest.skip("FIXME test not working")
    def test_write_place_holder(self):
        images = [PlaceHolder((20, 10, 1), 'float32'), PlaceHolder((20, 10, 1), 'float32')]
        out = '/tmp/bar.tif'
        Tiff.write(images, io=out)

        with Tiff.from_file(out) as tif:
            for img in tif:
                arr = img.memmap()
                arr[0,0] = 99
                arr[0,1] = 200

    # @unittest.skip("FIXME test not working")
    def test_write_place_holder_fast(self):
        '''Should only run on a Linux system with ext4 or XFS filesystem'''
        images = [PlaceHolder((20000, 10000, 1), 'uint8')]
        out = '/tmp/bar2.tif'

        start = time.time()

        Tiff.write(images, io=out)

        with Tiff.from_file(out) as tif:
            for img in tif:
                arr = img.memmap()
                arr[0,0] = 99
                arr[9999,9999] = 200

        end = time.time()
        assert end - start < 1 # sec

    @unittest.skip("FIXME test not working")
    def test_write(self):
        with Tiff.from_file(FILENAME) as tif:
            io = Tiff.write([np.atleast_3d([1]).astype('uint8'), np.atleast_3d([255]).astype('uint8')], closefd=False)
            actual = io.getvalue()
        with open('/tmp/foo1.tif', 'bw') as f:
            f.write(actual)

        expected = b'II+\x00\x08\x00\x00\x00' # header
        expected += b'\x10\x00\x00\x00\x00\x00\x00\x00' # first ifd position
        expected += b'\x0b\x00\x00\x00\x00\x00\x00\x00' # 13 entries
        expected += b'\x00\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x16\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x01\x10\x00\x01\x00\x00\x00\x00\x00\x00\x00\xfc\x00\x00\x00\x00\x00\x00\x00\xfd\x00\x00\x00\x00\x00\x00\x00\x01\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x16\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x01\x10\x00\x01\x00\x00\x00\x00\x00\x00\x00\xe9\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff'

        np.testing.assert_array_equal(actual, expected)

    def test_memmap_tcz(self):
        expected_z0 = \
         np.array([[91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91,  0, 91],
                   [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                  dtype='int')

        expected_z1 = \
         np.array([[0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0],
                   [0, 91,  0,  0, 91,  0,  0, 91,  0,  0, 91,  0,  0,  0,  0]],
                   dtype='int')

        fname = os.path.join(os.path.dirname(__file__),
                             'test_data/x15_y10_z2_c1.tif')

        m = Tiff.memmap_tcz(fname)

        np.testing.assert_array_equal(m[0, 0, 0], expected_z0)
        np.testing.assert_array_equal(m[0, 0, 1], expected_z1)
        assert isinstance(m[0, 0, 0], np.core.memmap)
        assert isinstance(m[0, 0, 1], np.core.memmap)

    def test_memmap_tcz_big(self):
        fname = os.path.join(os.path.dirname(__file__),
                             'test_data/x1_y1_c10_z11_t12.tif')
        m = Tiff.memmap_tcz(fname)
        print(m[0].shape)
        np.testing.assert_equal(len(m), 12)
        np.testing.assert_array_equal(m[0].shape, (10,11))

if __name__ == '__main__':
    unittest.main()
