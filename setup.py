from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='bigtiff',
    version='0.1.0',
    description='A library for random access for large (Big-)TIFF files',
    author='Manuel Schölling',
    author_email='manuel.schoelling@dzne.de',
    license='GPL2',
    packages=['bigtiff'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'bigtiff_lzw_decompress @ git+http://animate-x3.dzne.ds/idaf/lzw-decompressor.git@master',
        'kaitaistruct==0.8.0'
    ],
    dependency_links=[
      'git+http://animate-x3.dzne.ds/idaf/lzw-decompressor.git@master#egg=bigtiff_lzw_decompress-0.0.0',
    ]
)
