import time
import unittest
import os
import numpy as np
import shutil
from bigtiff import Tiff, PlaceHolder
from numpy import memmap as my_memmap
base_path = os.path.dirname(__file__)
FILENAME = os.path.join(os.path.dirname(__file__), 'foo.tif')


class TestTiff(unittest.TestCase):

    def test_memmap(self):
        FILENAME = os.path.join(os.path.dirname(__file__),
                                'test_data/x15_y10_z2_c1.tif')
        with Tiff.from_file(FILENAME) as tif:
            for img in tif:
                img.memmap()

    def test_load(self):
        FILENAME = os.path.join(os.path.dirname(__file__),
                                'test_data/x15_y10_z2_c1.tif')
        with Tiff.from_file(FILENAME) as tif:
            for img in tif:
                img.tags

    def test_write_place_holder(self):
        images = [PlaceHolder((20, 10, 1), 'float32'), PlaceHolder((20, 10, 1),
                  'float32')]
        out = '/tmp/bar.tif'
        Tiff.write(images, io=out)

        with Tiff.from_file(out) as tif:
            for img in tif:
                arr = img.memmap()
                arr[0, 0] = 99
                arr[0, 1] = 200

    def test_write_pixels(self):
        savedir = os.path.join(base_path, 'tmp')
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)
        images = [PlaceHolder((20, 10, 1), 'uint8'), PlaceHolder((20, 10, 1),
                  'float32')]
        path = os.path.join(savedir, 'test_write_pixels.tif')
        Tiff.write(images, io=path)

        with Tiff.from_file(path) as tif:
                slices = [img.memmap() for img in tif]

        assert slices[0].shape == (20, 10)
        assert slices[1].shape == (20, 10)
        slices[0][:, :5] = 255
        assert np.all(slices[0][:, :5] == 255)
        print('done')

        slices = None
        import gc
        gc.collect()
        print('end')

        with Tiff.from_file(path) as tif:
                slices = [img.memmap() for img in tif]
        assert (slices[0][:, :5] == 255).all()
        assert (slices[0][:, 5:] == 0).all()
        assert (slices[1] == 0).all()

    def test_write_pixelstack(self):
        savedir = os.path.join(base_path, 'tmp')
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)

        n_slices = 6
        images = [PlaceHolder((20, 10, 1), 'uint8') for _ in range(n_slices)]
        path = os.path.join(savedir, 'test_write_pixels.tif')
        Tiff.write(images, io=path)

        with Tiff.from_file(path) as tif:
                slices = [img.memmap() for img in tif]

        for slice in slices:
            assert np.all(slice == 0)

        slices[0][:, :5] = 10
        slices[1][:, :5] = 20
        slices[2][:, :5] = 30
        slices[3][:, :5] = 40
        slices[4][:, :5] = 50

        slices = None
        with Tiff.from_file(path) as tif:
                slices = [img.memmap() for img in tif]

        assert (slices[0][:, :5] == 10).all()
        assert (slices[0][:, 5:] == 0).all()

        assert (slices[1][:, :5] == 20).all()
        assert (slices[1][:, 5:] == 0).all()

        assert (slices[2][:, :5] == 30).all()
        assert (slices[2][:, 5:] == 0).all()

        assert (slices[3][:, :5] == 40).all()
        assert (slices[3][:, 5:] == 0).all()

        assert (slices[4][:, :5] == 50).all()
        assert (slices[4][:, 5:] == 0).all()

        assert (slices[5] == 0).all()





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
        assert isinstance(m[0, 0, 0], my_memmap)
        assert isinstance(m[0, 0, 1], my_memmap)

    def test_memmap_tcz_big(self):
        fname = os.path.join(os.path.dirname(__file__),
                             'test_data/x1_y1_c10_z11_t12.tif')
        m = Tiff.memmap_tcz(fname)
        print(m[0].shape)
        np.testing.assert_equal(len(m), 12)
        np.testing.assert_array_equal(m[0].shape, (10,11))

if __name__ == '__main__':
    unittest.main()
