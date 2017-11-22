BigTiff
=======

A library for random access on (Big-) Tiff files.


Examples
--------

```python
# create a huge empty image
images = [PlaceHolder((20000, 10000, 1), 'uint8')]
fname = '/tmp/myimage.tif'
Tiff.write(images, io=fname)

# assign a pixels
with Tiff.from_file(fname) as tif:
    for img in tif:
        print(img.tags)
        arr = img.memmap()
        arr[5000,5000,0] = 99
```


Installation
------------

Run `pip3 install 'http://animate-x3.dzne.ds/schoellingm/tiff/repository/master/archive.zip' --trusted-host animate-x3.dzne.ds --process-dependency-links`

