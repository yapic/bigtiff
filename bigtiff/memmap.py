import ctypes.util
from mmap import MAP_SHARED
import os

INVALID_POINTER = ctypes.c_void_p(-1).value

def check_mmap(addr, func, args):
    errno = ctypes.get_errno()
    err = os.strerror(errno)
    assert addr != INVALID_POINTER, 'mmap() failed: {}'.format(err)
    return addr


def check_munmap(ret, func, args):
    errno = ctypes.get_errno()
    assert ret != -1, 'munmap() failed: {}'.format(os.strerror(errno))


libc = ctypes.util.find_library('c')
libc = ctypes.CDLL(libc)

libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_size_t)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.errcheck = check_mmap

libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
libc.munmap.restype = ctypes.c_int
libc.munmap.errcheck = check_munmap

from numpy.core import memmap as np_memmap
import unittest.mock

def memmap(*args, **kwds):
    '''
    Works just like the ordinary numpy.memmap() but does not
    open an additional file descriptor.
    '''
    with unittest.mock.patch('mmap.mmap', new=mmap):
        arr = np_memmap(*args, **kwds)
    return arr


def mmap(fileno, bytelen, access=None, offset=0):
    '''
    Works just like the ordinary mmap.mmap() but does not
    open an additional file descriptor.
    '''
    buffer_type = ctypes.c_char * bytelen
    class MMap(buffer_type):
        def __del__(self):
            libc.munmap(addr, bytelen)

    addr = libc.mmap(None, bytelen, access, MAP_SHARED, fileno, offset)
    return MMap.from_address(addr)

