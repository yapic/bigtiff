import tif_format

from image2d import Image2dIterator
import representation

class Tiff(tif_format.TifFormat):
    def __iter__(self):
        if self.header.is_big_tiff is None:
            return Image2dIterator(self.header.first_ifd)
        else:
            return Image2dIterator(self.header.first_ifd_big_tiff)


def main():
    import sys
    fname = sys.argv[1]

    tif = Tiff.from_file(fname)
    for img in tif:
        print(img.tags)
        print(img[65:100,5:101,:].shape)


if __name__ == '__main__':
    main()

