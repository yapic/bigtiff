import bigtiff.tif_format as tif_format

tif_format.TifFormat.URatio.__repr__ = lambda self: repr(float(self.nominator)/self.denominator)
tif_format.TifFormat.SRatio.__repr__ = lambda self: repr(float(self.nominator)/self.denominator)
tif_format.TifFormat.String.__repr__ = lambda self: repr(self.string)

