# ovs128-rho0.8 accuracy test data

Copied from the isce3 repository (`tests/data/ampcor/accuracy-testdata/ovs128-rho0.8`,
last updated in isce3 #194).

* `img1_WN_512x512_1x1_128`, `img2_WN_512x512_1x1_128`: a pair of 512x512
  single-precision complex (complex64, little-endian, no header) white-noise
  (WN) images; `ovs128` and `rho0.8` in the name refer to their generation
  parameters (oversampling, correlation). The estimated offsets are close to
  zero. The `.vrt` files describe the raw layout for GDAL.
* `golden_isce3/`: outputs of the isce3 (v1) pycuampcor, two-pass workflow,
  used by the isce3 test `tests/python/packages/isce3/matchtemplate/test_ampcor.py`.
  `gross_offsets` is an empty (all zero) file created by the isce3 test.
* `golden/`: reference outputs of this package for all workflows and correlation
  surface oversampling methods, generated with `tests/python/make_golden.py`.

The parameters (see `tests/python/test_regression.py`): window 64 (across) x 32 (down),
half search range 20, skip 32, deramp method 1 along azimuth, raw data
oversampling 2, correlation surface zoom-in window 8 oversampled by 64,
statistics window 21, frequency-domain correlation, chunks of 1 x 2 windows.
