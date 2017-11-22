# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class TifFormat(KaitaiStruct):

    class Endian(Enum):
        le = 18761
        be = 19789

    class TagType(Enum):
        u1 = 1
        string = 2
        u2 = 3
        u4 = 4
        u_ratio = 5
        s1 = 6
        undefined = 7
        s2 = 8
        s4 = 9
        s_ratio = 10
        f4 = 11
        f8 = 12
        u8 = 16
        s8 = 17
        ifd8 = 18

    class Tag(Enum):
        new_subfile_type = 254
        subfile_type = 255
        image_width = 256
        image_length = 257
        bits_per_sample = 258
        compression = 259
        photometric_interpretation = 262
        threshholding = 263
        cell_width = 264
        cell_length = 265
        fill_order = 266
        document_name = 269
        image_description = 270
        make = 271
        model = 272
        strip_offsets = 273
        orientation = 274
        samples_per_pixel = 277
        rows_per_strip = 278
        strip_byte_counts = 279
        min_sample_value = 280
        max_sample_value = 281
        x_resolution = 282
        y_resolution = 283
        planar_configuration = 284
        page_name = 285
        x_position = 286
        y_position = 287
        free_offsets = 288
        free_byte_counts = 289
        gray_response_unit = 290
        gray_response_curve = 291
        t4_options = 292
        t6_options = 293
        resolution_unit = 296
        page_number = 297
        transfer_function = 301
        software = 305
        date_time = 306
        artist = 315
        host_computer = 316
        predictor = 317
        white_point = 318
        primary_chromaticities = 319
        color_map = 320
        halftone_hints = 321
        tile_width = 322
        tile_length = 323
        tile_offsets = 324
        tile_byte_counts = 325
        bad_fax_lines = 326
        clean_fax_data = 327
        consecutive_bad_fax_lines = 328
        sub_if_ds = 330
        ink_set = 332
        ink_names = 333
        number_of_inks = 334
        dot_range = 336
        target_printer = 337
        extra_samples = 338
        sample_format = 339
        s_min_sample_value = 340
        s_max_sample_value = 341
        transfer_range = 342
        clip_path = 343
        x_clip_path_units = 344
        y_clip_path_units = 345
        indexed = 346
        jpeg_tables = 347
        opi_proxy = 351
        global_parameters_ifd = 400
        profile_type = 401
        fax_profile = 402
        coding_methods = 403
        version_year = 404
        mode_number = 405
        decode = 433
        default_image_color = 434
        jpeg_proc = 512
        jpeg_interchange_format = 513
        jpeg_interchange_format_length = 514
        jpeg_restart_interval = 515
        jpeg_lossless_predictors = 517
        jpeg_point_transforms = 518
        jpegq_tables = 519
        jpegdc_tables = 520
        jpegac_tables = 521
        y_cb_cr_coefficients = 529
        y_cb_cr_sub_sampling = 530
        y_cb_cr_positioning = 531
        reference_black_white = 532
        strip_row_counts = 559
        xmp = 700
        image_id = 32781
        copyright = 33432
        image_layer = 34732
        ij_metadata_byte_counts = 50838
        ij_metadata = 50839
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.endian = self._root.Endian(self._io.read_u2be())
        self.header = self._root.Header(self._io, self, self._root)

    class Ifd(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            _on = self._root.header.is_big_tiff
            if _on == True:
                self.count = self._io.read_u8le()
            elif _on == False:
                self.count = self._io.read_u2le()
            self.entries = [None] * (self.count)
            for i in range(self.count):
                self.entries[i] = self._root.IfdEntry(self._io, self, self._root)

            self.next_ifd_position = self._root.Offset(self._io, self, self._root)

        def _read_be(self):
            _on = self._root.header.is_big_tiff
            if _on == True:
                self.count = self._io.read_u8be()
            elif _on == False:
                self.count = self._io.read_u2be()
            self.entries = [None] * (self.count)
            for i in range(self.count):
                self.entries[i] = self._root.IfdEntry(self._io, self, self._root)

            self.next_ifd_position = self._root.Offset(self._io, self, self._root)

        @property
        def next_ifd(self):
            if hasattr(self, '_m_next_ifd'):
                return self._m_next_ifd if hasattr(self, '_m_next_ifd') else None

            if self.next_ifd_position.value != 0:
                _pos = self._io.pos()
                self._io.seek(self.next_ifd_position.value)
                if self._is_le:
                    self._m_next_ifd = self._root.Ifd(self._io, self, self._root)
                else:
                    self._m_next_ifd = self._root.Ifd(self._io, self, self._root)
                self._io.seek(_pos)

            return self._m_next_ifd if hasattr(self, '_m_next_ifd') else None


    class TagValue(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            self.array = []
            while not self._io.is_eof():
                _on = self._parent.tag_type
                if _on == self._root.TagType.u_ratio:
                    self.array.append(self._root.URatio(self._io, self, self._root))
                elif _on == self._root.TagType.s2:
                    self.array.append(self._io.read_s2le())
                elif _on == self._root.TagType.s1:
                    self.array.append(self._io.read_s1())
                elif _on == self._root.TagType.u4:
                    self.array.append(self._io.read_u4le())
                elif _on == self._root.TagType.s4:
                    self.array.append(self._io.read_s4le())
                elif _on == self._root.TagType.u1:
                    self.array.append(self._io.read_u1())
                elif _on == self._root.TagType.f4:
                    self.array.append(self._io.read_f4le())
                elif _on == self._root.TagType.u2:
                    self.array.append(self._io.read_u2le())
                elif _on == self._root.TagType.undefined:
                    self.array.append(self._io.read_u1())
                elif _on == self._root.TagType.f8:
                    self.array.append(self._io.read_f8le())
                elif _on == self._root.TagType.s_ratio:
                    self.array.append(self._root.SRatio(self._io, self, self._root))
                elif _on == self._root.TagType.string:
                    self.array.append(self._root.String(self._io, self, self._root))
                else:
                    self.array.append(self._io.read_u8le())


        def _read_be(self):
            self.array = []
            while not self._io.is_eof():
                _on = self._parent.tag_type
                if _on == self._root.TagType.u_ratio:
                    self.array.append(self._root.URatio(self._io, self, self._root))
                elif _on == self._root.TagType.s2:
                    self.array.append(self._io.read_s2be())
                elif _on == self._root.TagType.s1:
                    self.array.append(self._io.read_s1())
                elif _on == self._root.TagType.u4:
                    self.array.append(self._io.read_u4be())
                elif _on == self._root.TagType.s4:
                    self.array.append(self._io.read_s4be())
                elif _on == self._root.TagType.u1:
                    self.array.append(self._io.read_u1())
                elif _on == self._root.TagType.f4:
                    self.array.append(self._io.read_f4be())
                elif _on == self._root.TagType.u2:
                    self.array.append(self._io.read_u2be())
                elif _on == self._root.TagType.undefined:
                    self.array.append(self._io.read_u1())
                elif _on == self._root.TagType.f8:
                    self.array.append(self._io.read_f8be())
                elif _on == self._root.TagType.s_ratio:
                    self.array.append(self._root.SRatio(self._io, self, self._root))
                elif _on == self._root.TagType.string:
                    self.array.append(self._root.String(self._io, self, self._root))
                else:
                    self.array.append(self._io.read_u8be())



    class String(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.string = (self._io.read_bytes_term(0, False, True, True)).decode(u"utf-8")


    class Offset(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            _on = self._root.header.is_big_tiff
            if _on == False:
                self.value = self._io.read_u4le()
            elif _on == True:
                self.value = self._io.read_u8le()

        def _read_be(self):
            _on = self._root.header.is_big_tiff
            if _on == False:
                self.value = self._io.read_u4be()
            elif _on == True:
                self.value = self._io.read_u8be()


    class Header(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            self.magic = self._io.read_u2le()
            if self.is_big_tiff:
                self.offset_size_raw = self._io.read_u2le()

            if not (self.is_big_tiff):
                self.first_ifd_position_u4 = self._io.read_u4le()

            if self.is_big_tiff:
                self.zeros = self._io.read_u2le()

            if self.is_big_tiff:
                self.first_ifd_position_big_tiff = self._io.read_u8le()


        def _read_be(self):
            self.magic = self._io.read_u2be()
            if self.is_big_tiff:
                self.offset_size_raw = self._io.read_u2be()

            if not (self.is_big_tiff):
                self.first_ifd_position_u4 = self._io.read_u4be()

            if self.is_big_tiff:
                self.zeros = self._io.read_u2be()

            if self.is_big_tiff:
                self.first_ifd_position_big_tiff = self._io.read_u8be()


        @property
        def is_big_tiff(self):
            if hasattr(self, '_m_is_big_tiff'):
                return self._m_is_big_tiff if hasattr(self, '_m_is_big_tiff') else None

            self._m_is_big_tiff = self.magic == 43
            return self._m_is_big_tiff if hasattr(self, '_m_is_big_tiff') else None

        @property
        def first_ifd(self):
            if hasattr(self, '_m_first_ifd'):
                return self._m_first_ifd if hasattr(self, '_m_first_ifd') else None

            if not (self.is_big_tiff):
                _pos = self._io.pos()
                self._io.seek(self.first_ifd_position_u4)
                if self._is_le:
                    self._m_first_ifd = self._root.Ifd(self._io, self, self._root)
                else:
                    self._m_first_ifd = self._root.Ifd(self._io, self, self._root)
                self._io.seek(_pos)

            return self._m_first_ifd if hasattr(self, '_m_first_ifd') else None

        @property
        def first_ifd_big_tiff(self):
            if hasattr(self, '_m_first_ifd_big_tiff'):
                return self._m_first_ifd_big_tiff if hasattr(self, '_m_first_ifd_big_tiff') else None

            if self.is_big_tiff:
                _pos = self._io.pos()
                self._io.seek(self.first_ifd_position_big_tiff)
                if self._is_le:
                    self._m_first_ifd_big_tiff = self._root.Ifd(self._io, self, self._root)
                else:
                    self._m_first_ifd_big_tiff = self._root.Ifd(self._io, self, self._root)
                self._io.seek(_pos)

            return self._m_first_ifd_big_tiff if hasattr(self, '_m_first_ifd_big_tiff') else None


    class SRatio(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            self.nominator = self._io.read_s4le()
            self.denominator = self._io.read_s4le()

        def _read_be(self):
            self.nominator = self._io.read_s4be()
            self.denominator = self._io.read_s4be()

        @property
        def value(self):
            if hasattr(self, '_m_value'):
                return self._m_value if hasattr(self, '_m_value') else None

            self._m_value = ((1.0 * self.nominator) / self.denominator)
            return self._m_value if hasattr(self, '_m_value') else None


    class IfdEntry(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            self.tag_raw = self._io.read_u2le()
            self.tag_type_raw = self._io.read_u2le()
            _on = self._root.header.is_big_tiff
            if _on == False:
                self.count = self._io.read_u4le()
            elif _on == True:
                self.count = self._io.read_u8le()
            if self.is_value:
                self._raw_values = self._io.read_bytes((self.tag_type_length * self.count))
                io = KaitaiStream(BytesIO(self._raw_values))
                self.values = self._root.TagValue(io, self, self._root)

            if self.is_value:
                self.padding = self._io.read_bytes((self.value_size - (self.tag_type_length * self.count)))

            if not (self.is_value):
                self.offset = self._root.Offset(self._io, self, self._root)


        def _read_be(self):
            self.tag_raw = self._io.read_u2be()
            self.tag_type_raw = self._io.read_u2be()
            _on = self._root.header.is_big_tiff
            if _on == False:
                self.count = self._io.read_u4be()
            elif _on == True:
                self.count = self._io.read_u8be()
            if self.is_value:
                self._raw_values = self._io.read_bytes((self.tag_type_length * self.count))
                io = KaitaiStream(BytesIO(self._raw_values))
                self.values = self._root.TagValue(io, self, self._root)

            if self.is_value:
                self.padding = self._io.read_bytes((self.value_size - (self.tag_type_length * self.count)))

            if not (self.is_value):
                self.offset = self._root.Offset(self._io, self, self._root)


        @property
        def is_value(self):
            if hasattr(self, '_m_is_value'):
                return self._m_is_value if hasattr(self, '_m_is_value') else None

            self._m_is_value = (self.tag_type_length * self.count) <= self.value_size
            return self._m_is_value if hasattr(self, '_m_is_value') else None

        @property
        def value_size(self):
            if hasattr(self, '_m_value_size'):
                return self._m_value_size if hasattr(self, '_m_value_size') else None

            self._m_value_size = (4 + ((int(self._root.header.is_big_tiff) & 1) * 4))
            return self._m_value_size if hasattr(self, '_m_value_size') else None

        @property
        def tag(self):
            if hasattr(self, '_m_tag'):
                return self._m_tag if hasattr(self, '_m_tag') else None

            self._m_tag = self._root.Tag(self.tag_raw)
            return self._m_tag if hasattr(self, '_m_tag') else None

        @property
        def tag_type_length(self):
            if hasattr(self, '_m_tag_type_length'):
                return self._m_tag_type_length if hasattr(self, '_m_tag_type_length') else None

            self._m_tag_type_length = (1 if self.tag_type == self._root.TagType.u1 else (1 if self.tag_type == self._root.TagType.string else (2 if self.tag_type == self._root.TagType.u2 else (4 if self.tag_type == self._root.TagType.u4 else (8 if self.tag_type == self._root.TagType.u8 else (8 if self.tag_type == self._root.TagType.u_ratio else (1 if self.tag_type == self._root.TagType.s1 else (1 if self.tag_type == self._root.TagType.undefined else (2 if self.tag_type == self._root.TagType.s2 else (4 if self.tag_type == self._root.TagType.s4 else (8 if self.tag_type == self._root.TagType.s8 else (8 if self.tag_type == self._root.TagType.s_ratio else (4 if self.tag_type == self._root.TagType.f4 else (8 if self.tag_type == self._root.TagType.f8 else 999999999))))))))))))))
            return self._m_tag_type_length if hasattr(self, '_m_tag_type_length') else None

        @property
        def tag_type(self):
            if hasattr(self, '_m_tag_type'):
                return self._m_tag_type if hasattr(self, '_m_tag_type') else None

            self._m_tag_type = self._root.TagType(self.tag_type_raw)
            return self._m_tag_type if hasattr(self, '_m_tag_type') else None

        @property
        def external_values(self):
            if hasattr(self, '_m_external_values'):
                return self._m_external_values if hasattr(self, '_m_external_values') else None

            _pos = self._io.pos()
            self._io.seek(self.offset.value)
            if self._is_le:
                self._raw__m_external_values = self._io.read_bytes((self.tag_type_length * self.count))
                io = KaitaiStream(BytesIO(self._raw__m_external_values))
                self._m_external_values = self._root.TagValue(io, self, self._root)
            else:
                self._raw__m_external_values = self._io.read_bytes((self.tag_type_length * self.count))
                io = KaitaiStream(BytesIO(self._raw__m_external_values))
                self._m_external_values = self._root.TagValue(io, self, self._root)
            self._io.seek(_pos)
            return self._m_external_values if hasattr(self, '_m_external_values') else None


    class URatio(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            _on = self._root.endian
            if _on == self._root.Endian.le:
                self._is_le = True
            elif _on == self._root.Endian.be:
                self._is_le = False

            if self._is_le == True:
                self._read_le()
            elif self._is_le == False:
                self._read_be()
            else:
                raise Exception("Unable to decide endianness")

        def _read_le(self):
            self.nominator = self._io.read_u4le()
            self.denominator = self._io.read_u4le()

        def _read_be(self):
            self.nominator = self._io.read_u4be()
            self.denominator = self._io.read_u4be()

        @property
        def value(self):
            if hasattr(self, '_m_value'):
                return self._m_value if hasattr(self, '_m_value') else None

            self._m_value = ((1.0 * self.nominator) / self.denominator)
            return self._m_value if hasattr(self, '_m_value') else None



