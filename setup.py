from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='bigtiff',
    version='0.1',
    description='A library for random access for large (Big-)TIFF files',
    author='Manuel Schölling',
    author_email='manuel.schoelling@dzne.de',
    license='GPL2',
    packages=['bigtiff'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'bigtiff_lzw_decompress',
        'kaitaistruct==0.7.99'
    ],
    dependency_links=[
      'git+http://animate-x3.dzne.ds/schoellingm/lzw-decompressor.git@master#egg=bigtiff_lzw_decompress-0.0.0',
      'git+https://github.com/kaitai-io/kaitai_struct_python_runtime.git@fe935e58eb03bdcbb0186deaac8631912b8fa513#egg=kaitaistruct-0.7.99',
    ]
)
