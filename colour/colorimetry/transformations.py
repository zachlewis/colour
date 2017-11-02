#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colour Matching Functions Transformations
=========================================

Defines various educational objects for colour matching functions
transformations:

-   :func:`RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
-   :func:`RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
-   :func:`LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
-   :func:`LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`

See Also
--------
`Colour Matching Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import LMS_CMFS, RGB_CMFS, PHOTOPIC_LEFS
from colour.utilities import dot_vector, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
    'RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
    'RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
    'LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
    'LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs'
]


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Wright & Guild 1931 2 Degree RGB CMFs* colour matching functions
    into the *CIE 1931 2 Degree Standard Observer* colour matching functions.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray
        *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1931 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [1]  Wyszecki, G., & Stiles, W. S. (2000). Table 1(3.3.3). In Color
            Science: Concepts and Methods, Quantitative Data and Formulae
            (pp. 138–139). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> from colour import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0113577...,  0.004102  ,  0.        ])
    """

    cmfs = RGB_CMFS['Wright & Guild 1931 2 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    rgb = rgb_bar / np.sum(rgb_bar)

    M1 = np.array(
        [[0.49000, 0.31000, 0.20000],
         [0.17697, 0.81240, 0.01063],
         [0.00000, 0.01000, 0.99000]])  # yapf: disable

    M2 = np.array(
        [[0.66697, 1.13240, 1.20063],
         [0.66697, 1.13240, 1.20063],
         [0.66697, 1.13240, 1.20063]])  # yapf: disable

    xyz = dot_vector(M1, rgb)
    xyz /= dot_vector(M2, rgb)

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    V = PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'].copy()
    V.align(cmfs.shape)
    L = V[wavelength]

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    xyz_bar = tstack((x_bar, y_bar, z_bar))

    return xyz_bar


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *CIE 1964 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray
        *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.

    See Also
    --------
    :attr:`colour.colorimetry.dataset.cmfs.RGB_CMFS`

    Notes
    -----
    -   Data for the *CIE 1964 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [2]  Wyszecki, G., & Stiles, W. S. (2000). The CIE 1964 Standard
            Observer. In Color Science: Concepts and Methods, Quantitative
            Data and Formulae (p. 141). Wiley. ISBN:978-0471399186

    Examples
    --------
    >>> from colour import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0096432...,  0.0037526..., -0.0000041...])
    """

    cmfs = RGB_CMFS['Stiles & Burch 1959 10 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    M = np.array(
        [[0.341080, 0.189145, 0.387529],
         [0.139058, 0.837460, 0.073316],
         [0.000000, 0.039553, 2.026200]])  # yapf: disable

    xyz_bar = dot_vector(M, rgb_bar)

    return xyz_bar


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *Stockman & Sharpe 10 Degree Cone Fundamentals*
    spectral sensitivity functions.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray
        *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral tristimulus
        values.

    Notes
    -----
    -   Data for the *Stockman & Sharpe 10 Degree Cone Fundamentals* already
        exists, this definition is intended for educational purpose.

    References
    ----------
    .. [3]  CIE TC 1-36. (2006). CIE 170-1:2006 Fundamental Chromaticity
            Diagram with Physiological Axes - Part 1 (pp. 1–56).
            ISBN:978-3-901-90646-6

    Examples
    --------
    >>> from colour import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0052860...,  0.0003252...,  0.        ])
    """

    cmfs = RGB_CMFS['Stiles & Burch 1959 10 Degree RGB CMFs']

    rgb_bar = cmfs[wavelength]

    M = np.array(
        [[0.1923252690, 0.749548882, 0.0675726702],
         [0.0192290085, 0.940908496, 0.113830196],
         [0.0000000000, 0.0105107859, 0.991427669]])  # yapf: disable

    lms_bar = dot_vector(M, rgb_bar)
    lms_bar[..., -1][np.asarray(np.asarray(wavelength) > 505)] = 0

    return lms_bar


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 2 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 2 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray
        *CIE 2012 2 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 2 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [4]  CVRL. (n.d.). CIE (2012) 2-deg XYZ “physiologically-relevant”
            colour matching functions. Retrieved June 25, 2014, from
            http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm

    Examples
    --------
    >>> from colour import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0109677...,  0.0041959...,  0.        ])
    """

    cmfs = LMS_CMFS['Stockman & Sharpe 2 Degree Cone Fundamentals']

    lms_bar = cmfs[wavelength]

    M = np.array(
        [[1.94735469, -1.41445123, 0.36476327],
         [0.68990272, 0.34832189, 0.00000000],
         [0.00000000, 0.00000000, 1.93485343]])  # yapf: disable

    xyz_bar = dot_vector(M, lms_bar)

    return xyz_bar


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 10 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 10 Degree Standard Observer* colour matching
    functions.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in nm.

    Returns
    -------
    ndarray
        *CIE 2012 10 Degree Standard Observer* spectral tristimulus values.

    Notes
    -----
    -   Data for the *CIE 2012 10 Degree Standard Observer* already exists,
        this definition is intended for educational purpose.

    References
    ----------
    .. [5]  CVRL. (n.d.). CIE (2012) 10-deg XYZ “physiologically-relevant”
            colour matching functions. Retrieved June 25, 2014, from
            http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm

    Examples
    --------
    >>> from colour import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)  # doctest: +ELLIPSIS
    array([ 0.0098162...,  0.0037761...,  0.        ])
    """

    cmfs = LMS_CMFS['Stockman & Sharpe 10 Degree Cone Fundamentals']

    lms_bar = cmfs[wavelength]

    M = np.array(
        [[1.93986443, -1.34664359, 0.43044935],
         [0.69283932, 0.34967567, 0.00000000],
         [0.00000000, 0.00000000, 2.14687945]])  # yapf: disable

    xyz_bar = dot_vector(M, lms_bar)

    return xyz_bar
