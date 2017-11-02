#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colour Matching Functions
=========================

Defines colour matching functions classes for the dataset from
:mod:`colour.colorimetry.dataset.cmfs` module:

-   :class:`LMS_ConeFundamentals`: Implements support for the
    Stockman and Sharpe *LMS* cone fundamentals colour matching functions.
-   :class:`RGB_ColourMatchingFunctions`: Implements support for the *CIE RGB*
    colour matching functions.
-   :class:`XYZ_ColourMatchingFunctions`: Implements support for the *CIE*
    Standard Observers *XYZ* colour matching functions.

See Also
--------
`Colour Matching Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
colour.colorimetry.dataset.cmfs,
colour.colorimetry.spectrum.MultiSpectralPowerDistribution
"""

from __future__ import division, unicode_literals

from colour.colorimetry import MultiSpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'LMS_ConeFundamentals', 'RGB_ColourMatchingFunctions',
    'XYZ_ColourMatchingFunctions'
]


class LMS_ConeFundamentals(MultiSpectralPowerDistribution):
    """
    Implements support for the Stockman and Sharpe *LMS* cone fundamentals
    colour matching functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignal or \
MultiSpectralPowerDistribution or array_like or dict_like, optional
        Data to be stored in the multi-spectral power distribution.
    domain : array_like, optional
        Values to initialise the multiple :class:`SpectralPowerDistribution`
        class instances :attr:`Signal.wavelengths` attribute with. If both
        `data` and `domain` arguments are defined, the latter with be used to
        initialise the :attr:`Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`SpectralPowerDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral power distribution name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`SpectralPowerDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`SpectralPowerDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`SpectralPowerDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`SpectralPowerDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral power distribution labels for figures, default to
        :attr:`LMS_ConeFundamentals.labels` attribute value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(LMS_ConeFundamentals, self).__init__(
            data,
            domain,
            labels=('l_bar', 'm_bar', 's_bar'),
            strict_labels=('$\\bar{l}$', '$\\bar{m}', '$\\bar{s}'),
            **kwargs)


class RGB_ColourMatchingFunctions(MultiSpectralPowerDistribution):
    """
    Implements support for the *CIE RGB* colour matching functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignal or \
MultiSpectralPowerDistribution or array_like or dict_like, optional
        Data to be stored in the multi-spectral power distribution.
    domain : array_like, optional
        Values to initialise the multiple :class:`SpectralPowerDistribution`
        class instances :attr:`Signal.wavelengths` attribute with. If both
        `data` and `domain` arguments are defined, the latter with be used to
        initialise the :attr:`Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`SpectralPowerDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral power distribution name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`SpectralPowerDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`SpectralPowerDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`SpectralPowerDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`SpectralPowerDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral power distribution labels for figures, default to
        :attr:`RGB_ColourMatchingFunctions.labels` attribute value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(RGB_ColourMatchingFunctions, self).__init__(
            data,
            domain,
            labels=('r_bar', 'g_bar', 'b_bar'),
            strict_labels=('$\\bar{r}$', '$\\bar{g}', '$\\bar{b}'),
            **kwargs)


class XYZ_ColourMatchingFunctions(MultiSpectralPowerDistribution):
    """
    Implements support for the *CIE* Standard Observers *XYZ* colour matching
    functions.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignal or \
MultiSpectralPowerDistribution or array_like or dict_like, optional
        Data to be stored in the multi-spectral power distribution.
    domain : array_like, optional
        Values to initialise the multiple :class:`SpectralPowerDistribution`
        class instances :attr:`Signal.wavelengths` attribute with. If both
        `data` and `domain` arguments are defined, the latter with be used to
        initialise the :attr:`Signal.wavelengths` attribute.
    labels : array_like, optional
        Names to use for the :class:`SpectralPowerDistribution` class
        instances.

    Other Parameters
    ----------------
    name : unicode, optional
       Multi-spectral power distribution name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`SpectralPowerDistribution` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`SpectralPowerDistribution` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`SpectralPowerDistribution` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`SpectralPowerDistribution` class instances.
    strict_labels : array_like, optional
        Multi-spectral power distribution labels for figures, default to
        :attr:`XYZ_ColourMatchingFunctions.labels` attribute value.
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(XYZ_ColourMatchingFunctions, self).__init__(
            data,
            domain,
            labels=('x_bar', 'y_bar', 'z_bar'),
            strict_labels=('$\\bar{x}$', '$\\bar{y}', '$\\bar{z}'),
            **kwargs)
