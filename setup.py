from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='bigtiff',
    version='0.1.2',
    description='A library for random access for large (Big-)TIFF files',
    author='Manuel Schölling',
    author_email='manuel.schoelling@dzne.de',
    license='GPL3',
    packages=['bigtiff'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'kaitaistruct==0.8.0'
    ],
)
