meta:
  ks-version: 0.8
  id: tif_format
  file-extension:
    - tif
    - tiff


seq:
  - id: endian
    type: u2be
    enum: endian
  - id: header
    type: header

types:
  header:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: magic
        type: u2
      - id: offset_size_raw
        if: is_big_tiff
        type: u2
      - id: first_ifd_position_u4
        if: not is_big_tiff
        type: u4
      - id: zeros
        if: is_big_tiff
        type: u2
      - id: first_ifd_position_big_tiff
        if: is_big_tiff
        type: u8

    instances:
      is_big_tiff:
        value: magic == 43
      first_ifd:
        pos: first_ifd_position_u4
        if: not is_big_tiff
        type: ifd
      first_ifd_big_tiff:
        if: is_big_tiff
        type: ifd
        pos: first_ifd_position_big_tiff

  offset:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: value
        type:
          switch-on: _root.header.is_big_tiff
          cases:
            false:  u4
            true:   u8

  ifd:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: count
        type:
          switch-on: _root.header.is_big_tiff
          cases:
            true: u8
            false: u2
      - id: entries
        type: ifd_entry
        repeat: expr
        repeat-expr: count
      - id: next_ifd_position
        type: offset
    instances:
      next_ifd:
        if: next_ifd_position.value != 0
        type: ifd
        pos: next_ifd_position.value

  ifd_entry:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: tag_raw
        type: u2
      - id: tag_type_raw
        type: u2
      - id: count
        type:
          switch-on: _root.header.is_big_tiff
          cases:
            false: u4
            true: u8
      - id: values
        size: tag_type_length * count
        type: tag_value
        if: is_value
      - id: padding
        size: value_size - tag_type_length * count
        if: is_value
      - id: offset
        type: offset
        if: not is_value
    instances:
      value_size:
        value: 4 + (_root.header.is_big_tiff.to_i & 1)*4
      tag:
        value: tag_raw
        enum: tag
      tag_type:
        value: tag_type_raw
        enum: tag_type
      tag_type_length:
        value: >
          (tag_type == tag_type::u1) ?        1 :
          (tag_type == tag_type::string) ?    1 :
          (tag_type == tag_type::u2) ?        2 :
          (tag_type == tag_type::u4) ?        4 :
          (tag_type == tag_type::u8) ?        8 :
          (tag_type == tag_type::u_ratio) ?   8 :
          (tag_type == tag_type::s1) ?        1 :
          (tag_type == tag_type::undefined) ? 1 :
          (tag_type == tag_type::s2) ?        2 :
          (tag_type == tag_type::s4) ?        4 :
          (tag_type == tag_type::s8) ?        8 :
          (tag_type == tag_type::s_ratio) ?   8 :
          (tag_type == tag_type::f4) ?        4 :
          (tag_type == tag_type::f8) ?        8 :
          999999999
      is_value:
        value: tag_type_length * count <= value_size
      external_values:
        pos: offset.value
        type: tag_value
        size: tag_type_length * count

  tag_value:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: array
        repeat: eos
        type:
          switch-on: _parent.tag_type
          cases:
            'tag_type::u1': u1
            'tag_type::string': string
            'tag_type::u4': u4
            'tag_type::u2': u2
            'tag_type::u4': u4
            'tag_type::u_ratio': u_ratio
            'tag_type::s1': s1
            'tag_type::undefined': u1
            'tag_type::s2': s2
            'tag_type::s4': s4
            'tag_type::s_ratio': s_ratio
            'tag_type::f4': f4
            'tag_type::f8': f8
            _: u8
  string:
    seq:
      - id: string
        type: strz
        encoding: utf-8

  u_ratio:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: nominator
        type: u4
      - id: denominator
        type: u4
    instances:
      value:
        value: 1.0*nominator/denominator

  s_ratio:
    meta:
      endian:
        switch-on: _root.endian
        cases:
          'endian::le': le
          'endian::be': be
    seq:
      - id: nominator
        type: s4
      - id: denominator
        type: s4
    instances:
      value:
        value: 1.0*nominator/denominator

enums:
  endian:
    0x4949: le
    0x4d4d: be

  tag_type:
    0x0001: u1
    0x0002: string
    0x0003: u2
    0x0004: u4
    0x0005: u_ratio
    0x0006: s1
    0x0007: undefined
    0x0008: s2
    0x0009: s4
    0x000A: s_ratio
    0x000B: f4
    0x000C: f8
    0x0010: u8 # http://www.awaresystems.be/imaging/tiff/bigtiff.html#other
    0x0011: s8 # http://www.awaresystems.be/imaging/tiff/bigtiff.html#other
    0x0012: ifd8 # http://www.awaresystems.be/imaging/tiff/bigtiff.html#other

    # http://www.awaresystems.be/imaging/tiff/tifftags/extension.html

  tag:
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=17
    0x0106: photometric_interpretation
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=17
    0x0103: compression
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=18
    0x0101: image_length
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=18
    0x0100: image_width
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=18
    0x0128: resolution_unit
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=19
    0x011a: x_resolution
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=19
    0x011b: y_resolution
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=19
    0x0116: rows_per_strip
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=19
    0x0111: strip_offsets
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=19
    0x0117: strip_byte_counts
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=22
    0x0102: bits_per_sample
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=23
    0x0140: color_map
    # https://web.archive.org/web/20091223030231/http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf#page=24
    0x0115: samples_per_pixel

    # http://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
    # A general indication of the kind of data contained in this subfile.
    254: new_subfile_type
    # A general indication of the kind of data contained in this subfile.
    255: subfile_type
    # The number of columns in the image, i.e., the number of pixels per row.
    256: image_width
    # The number of rows of pixels in the image.
    257: image_length
    # Number of bits per component.
    258: bits_per_sample
    # Compression scheme used on the image data.
    259: compression
    # The color space of the image data.
    262: photometric_interpretation
    # For black and white TIFF files that represent shades of gray, the technique used to convert from gray to black and white pixels.
    263: threshholding
    # The width of the dithering or halftoning matrix used to create a dithered or halftoned bilevel file.
    264: cell_width
    # The length of the dithering or halftoning matrix used to create a dithered or halftoned bilevel file.
    265: cell_length
    # The logical order of bits within a byte.
    266: fill_order
    # A string that describes the subject of the image.
    270: image_description
    # The scanner manufacturer.
    271: make
    # The scanner model name or number.
    272: model
    # For each strip, the byte offset of that strip.
    273: strip_offsets
    # The orientation of the image with respect to the rows and columns.
    274: orientation
    # The number of components per pixel.
    277: samples_per_pixel
    # The number of rows per strip.
    278: rows_per_strip
    # For each strip, the number of bytes in the strip after compression.
    279: strip_byte_counts
    # The minimum component value used.
    280: min_sample_value
    # The maximum component value used.
    281: max_sample_value
    # The number of pixels per ResolutionUnit in the ImageWidth direction.
    282: x_resolution
    # The number of pixels per ResolutionUnit in the ImageLength direction.
    283: y_resolution
    # How the components of each pixel are stored.
    284: planar_configuration
    # For each string of contiguous unused bytes in a TIFF file, the byte offset of the string.
    288: free_offsets
    # For each string of contiguous unused bytes in a TIFF file, the number of bytes in the string.
    289: free_byte_counts
    # The precision of the information contained in the GrayResponseCurve.
    290: gray_response_unit
    # For grayscale data, the optical density of each possible pixel value.
    291: gray_response_curve
    # The unit of measurement for XResolution and YResolution.
    296: resolution_unit
    # Name and version number of the software package(s) used to create the image.
    305: software
    # Date and time of image creation.
    306: date_time
    # Person who created the image.
    315: artist
    # The computer and/or operating system in use at the time of image creation.
    316: host_computer
    # A color map for palette color images.
    320: color_map
    # Description of extra components.
    338: extra_samples
    # Copyright notice.
    33432: copyright

    # http://www.awaresystems.be/imaging/tiff/tifftags/extension.html
    # The name of the document from which this image was scanned.
    269: document_name
    # The name of the page from which this image was scanned.
    285: page_name
    # X position of the image.
    286: x_position
    # Y position of the image.
    287: y_position
    # Options for Group 3 Fax compression
    292: t4_options
    # Options for Group 4 Fax compression
    293: t6_options
    # The page number of the page from which this image was scanned.
    297: page_number
    # Describes a transfer function for the image in tabular style.
    301: transfer_function
    # A mathematical operator that is applied to the image data before an encoding scheme is applied.
    317: predictor
    # The chromaticity of the white point of the image.
    318: white_point
    # The chromaticities of the primaries of the image.
    319: primary_chromaticities
    # Conveys to the halftone function the range of gray levels within a colorimetrically-specified image that should retain tonal detail.
    321: halftone_hints
    # The tile width in pixels. This is the number of columns in each tile.
    322: tile_width
    # The tile length (height) in pixels. This is the number of rows in each tile.
    323: tile_length
    # For each tile, the byte offset of that tile, as compressed and stored on disk.
    324: tile_offsets
    # For each tile, the number of (compressed) bytes in that tile.
    325: tile_byte_counts
    # Used in the TIFF-F standard, denotes the number of 'bad' scan lines encountered by the facsimile device.
    326: bad_fax_lines
    # Used in the TIFF-F standard, indicates if 'bad' lines encountered during reception are stored in the data, or if 'bad' lines have been replaced by the receiver.
    327: clean_fax_data
    # Used in the TIFF-F standard, denotes the maximum number of consecutive 'bad' scanlines received.
    328: consecutive_bad_fax_lines
    # Offset to child IFDs.
    330: sub_if_ds
    # The set of inks used in a separated (PhotometricInterpretation=5) image.
    332: ink_set
    # The name of each ink used in a separated image.
    333: ink_names
    # The number of inks.
    334: number_of_inks
    # The component values that correspond to a 0% dot and 100% dot.
    336: dot_range
    # A description of the printing environment for which this separation is intended.
    337: target_printer
    # Specifies how to interpret each data sample in a pixel.
    339: sample_format
    # Specifies the minimum sample value.
    340: s_min_sample_value
    # Specifies the maximum sample value.
    341: s_max_sample_value
    # Expands the range of the TransferFunction.
    342: transfer_range
    # Mirrors the essentials of PostScript's path creation functionality.
    343: clip_path
    # The number of units that span the width of the image, in terms of integer ClipPath coordinates.
    344: x_clip_path_units
    # The number of units that span the height of the image, in terms of integer ClipPath coordinates.
    345: y_clip_path_units
    # Aims to broaden the support for indexed images to include support for any color space.
    346: indexed
    # JPEG quantization and/or Huffman tables.
    347: jpeg_tables
    # OPI-related.
    351: opi_proxy
    # Used in the TIFF-FX standard to point to an IFD containing tags that are globally applicable to the complete TIFF file.
    400: global_parameters_ifd
    # Used in the TIFF-FX standard, denotes the type of data stored in this file or IFD.
    401: profile_type
    # Used in the TIFF-FX standard, denotes the 'profile' that applies to this file.
    402: fax_profile
    # Used in the TIFF-FX standard, indicates which coding methods are used in the file.
    403: coding_methods
    # Used in the TIFF-FX standard, denotes the year of the standard specified by the FaxProfile field.
    404: version_year
    # Used in the TIFF-FX standard, denotes the mode of the standard specified by the FaxProfile field.
    405: mode_number
    # Used in the TIFF-F and TIFF-FX standards, holds information about the ITULAB (PhotometricInterpretation = 10) encoding.
    433: decode
    # Defined in the Mixed Raster Content part of RFC 2301, is the default color needed in areas where no image is available.
    434: default_image_color
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    512: jpeg_proc
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    513: jpeg_interchange_format
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    514: jpeg_interchange_format_length
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    515: jpeg_restart_interval
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    517: jpeg_lossless_predictors
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    518: jpeg_point_transforms
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    519: jpegq_tables
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    520: jpegdc_tables
    # Old-style JPEG compression field. TechNote2 invalidates this part of the specification.
    521: jpegac_tables
    # The transformation from RGB to YCbCr image data.
    529: y_cb_cr_coefficients
    # Specifies the subsampling factors used for the chrominance components of a YCbCr image.
    530: y_cb_cr_sub_sampling
    # Specifies the positioning of subsampled chrominance components relative to luminance samples.
    531: y_cb_cr_positioning
    # Specifies a pair of headroom and footroom image data values (codes) for each pixel component.
    532: reference_black_white
    # Defined in the Mixed Raster Content part of RFC 2301, used to replace RowsPerStrip for IFDs with variable-sized strips.
    559: strip_row_counts
    # XML packet containing XMP metadata
    700: xmp
    # OPI-related.
    32781: image_id
    # Defined in the Mixed Raster Content part of RFC 2301, used to denote the particular function of this Image in the mixed raster scheme.
    34732: image_layer

    50839: ij_metadata
    50838: ij_metadata_byte_counts

