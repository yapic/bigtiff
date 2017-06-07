all:
	cd kaitai_struct/compiler/; sbt 'compilerJVM/run -t python ../../tif_format.ksy' && mv tif_format.py ../..; cd -
