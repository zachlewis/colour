#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interpolation
=============

Defines classes for interpolating variables.

-   :class:`KernelInterpolator`: 1-D function generic interpolation with
    arbitrary kernel.
-   :class:`LinearInterpolator`: 1-D function linear interpolation.
-   :class:`SpragueInterpolator`: 1-D function fifth-order polynomial
    interpolation using *Sprague (1880)* method.
-   :class:`CubicSplineInterpolator`: 1-D function cubic spline interpolation.
-   :class:`PchipInterpolator`: 1-D function piecewise cube Hermite
    interpolation.
-   :class:`NullInterpolator`: 1-D function null interpolation.
-   :func:`lagrange_coefficients`: Computation of *Lagrange Coefficients*.
"""

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate
from collections import OrderedDict
from six.moves import reduce

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import (as_numeric, interval, is_integer, is_numeric,
                              closest_indexes, warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'kernel_nearest_neighbour', 'kernel_linear', 'kernel_sinc',
    'kernel_lanczos', 'kernel_cardinal_spline', 'KernelInterpolator',
    'LinearInterpolator', 'SpragueInterpolator', 'CubicSplineInterpolator',
    'PchipInterpolator', 'NullInterpolator', 'lagrange_coefficients'
]


def kernel_nearest_neighbour(x):
    """
    Returns the *nearest-neighbour* kernel evaluated at given samples.

    Parameters
    ----------
    x : array_like
        Samples at which to evaluate the *nearest-neighbour* kernel.

    Returns
    -------
    ndarray
        The *nearest-neighbour* kernel evaluated at given samples.

    References
    ----------
    .. [1]  Burger, W., & Burge, M. J. (2009). Principles of Digital Image
            Processing. London: Springer London.
            https://doi.org/10.1007/978-1-84800-195-4

    Examples
    --------
    >>> kernel_nearest_neighbour(np.linspace(0, 1, 10))
    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    """

    return np.where(np.abs(x) < 0.5, 1, 0)


def kernel_linear(x):
    """
    Returns the *linear* kernel evaluated at given samples. [1]_

    Parameters
    ----------
    x : array_like
        Samples at which to evaluate the *linear* kernel.

    Returns
    -------
    ndarray
        The *linear* kernel evaluated at given samples.

    Examples
    --------
    >>> kernel_linear(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([ 1.        ,  0.8888888...,  0.7777777...,  \
0.6666666...,  0.5555555...,
            0.4444444...,  0.3333333...,  0.2222222...,  \
0.1111111...,  0.        ])
    """

    return np.where(np.abs(x) < 1, 1 - np.abs(x), 0)


def kernel_sinc(x, a=3):
    """
    Returns the *sinc* kernel evaluated at given samples. [1]_

    Parameters
    ----------
    x : array_like
        Samples at which to evaluate the *sinc* kernel.
    a : int, optional
        Size of the *sinc* kernel.

    Returns
    -------
    ndarray
        The *sinc* kernel evaluated at given samples.

    Examples
    --------
    >>> kernel_sinc(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([  1.0000000...e+00,   9.7981553...e-01,   9.2072542...e-01,
             8.2699334...e-01,   7.0531659...e-01,   5.6425327...e-01,
             4.1349667...e-01,   2.6306440...e-01,   1.2247694...e-01,
             3.8981718...e-17])
    """

    assert a >= 1, '"a" must be equal or superior to 1!'

    return np.where(np.abs(x) < a, np.sinc(x), 0)


def kernel_lanczos(x, a=3):
    """
    Returns the *lanczos* kernel evaluated at given samples.

    Parameters
    ----------
    x : array_like
        Samples at which to evaluate the *lanczos* kernel.
    a : int, optional
        Size of the *lanczos* kernel.

    Returns
    -------
    ndarray
        The *lanczos* kernel evaluated at given samples.

    References
    ----------
    .. [2]  Wikipedia. (n.d.). Lanczos resampling. Retrieved October 14, 2017,
            from https://en.wikipedia.org/wiki/Lanczos_resampling

    Examples
    --------
    >>> kernel_lanczos(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([  1.0000000...e+00,   9.7760615...e-01,   9.1243770...e-01,
             8.1030092...e-01,   6.8012706...e-01,   5.3295773...e-01,
             3.8071690...e-01,   2.3492839...e-01,   1.0554054...e-01,
             3.2237621...e-17])
    """

    assert a >= 1, '"a" must be equal or superior to 1!'

    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0)


def kernel_cardinal_spline(x, a=0.5, b=0.0):
    """
    Returns the *cardinal spline* kernel evaluated at given samples. [1]_

    Notable *cardinal spline* :math:`a` and :math:`b` parameterizations:

    -   *Catmull-Rom*: :math:`(a=0.5, b=0)`
    -   *Cubic B-Spline*: :math:`(a=0, b=1)`
    -   *Mitchell-Netravalli*: :math:`(a=\cfrac{1}{3}, b=\cfrac{1}{3})`

    Parameters
    ----------
    x : array_like
        Samples at which to evaluate the *cardinal spline* kernel.
    a : int, optional
        :math:`a` control parameter.
    b : int, optional
        :math:`b` control parameter.

    Returns
    -------
    ndarray
        The *cardinal spline* kernel evaluated at given samples.

    Examples
    --------
    >>> kernel_cardinal_spline(np.linspace(0, 1, 10))  # doctest: +ELLIPSIS
    array([ 1.        ,  0.9711934...,  0.8930041...,  \
0.7777777...,  0.6378600...,
            0.4855967...,  0.3333333...,  0.1934156...,  \
0.0781893...,  0.        ])
    """

    x_abs = np.abs(x)
    y = np.where(x_abs < 1, (-6 * a - 9 * b + 12) * x_abs ** 3 +
                 (6 * a + 12 * b - 18) * x_abs ** 2 - 2 * b + 6,
                 (-6 * a - b) * x_abs ** 3 + (30 * a + 6 * b) * x_abs ** 2 +
                 (-48 * a - 12 * b) * x_abs + 24 * a + 8 * b)
    y[x_abs >= 2] = 0

    return 1 / 6 * y


class KernelInterpolator(object):
    """
    Kernel based interpolation of a 1-D function. [2]_

    Parameters
    ----------
    x : array_like
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : array_like
        Dependent and already known :math:`y` variable values to
        interpolate.
    window : int, optional
        Width of the window in samples on each side.
    kernel : callable, optional
        Kernel to use for interpolation.
    kernel_args : dict, optional
         Arguments to use when calling the kernel.
    padding_args : dict, optional
         Arguments to use when padding :math:`y` variable values with the
         :func:`np.pad` definition.
    dtype : type
        Data type used for internal conversions.

    Attributes
    ----------
    x
    y
    window
    kernel
    kernel_args
    padding_args

    Methods
    -------
    __call__

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = KernelInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    6.9411400...

    Interpolating an *array_like* variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.1806208...,  8.0823848...])

    Using a different *lanczos* kernel:

    >>> f = KernelInterpolator(x, y, kernel=kernel_sinc)
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.5147317...,  8.3965466...])

    Using a different window size:

    >>> f = KernelInterpolator(
    ...     x,
    ...     y,
    ...     window=16,
    ...     kernel=kernel_lanczos,
    ...     kernel_args={'a': 16})
    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 5.396179...,  5.652109...])
    """

    def __init__(self,
                 x,
                 y,
                 window=3,
                 kernel=kernel_lanczos,
                 kernel_args=None,
                 padding_args=None,
                 dtype=DEFAULT_FLOAT_DTYPE):
        self._x_p = None
        self._y_p = None

        self._x = None
        self._y = None
        self._window = None
        self._padding_args = {'pad_width': (window, window), 'mode': 'reflect'}
        self._dtype = dtype

        self.x = x
        self.y = y
        self.window = window
        self.padding_args = padding_args

        self._kernel = None
        self.kernel = kernel
        self._kernel_args = {}
        self.kernel_args = kernel_args

        self._validate_dimensions()

    @property
    def x(self):
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        array_like
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value):
        """
        Setter for the **self.x** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

            value_interval = interval(value)

            if value_interval.size != 1:
                warning(('"x" independent variable is not uniform, '
                         'unpredictable results may occur!'))

            self._x = value

            if self._window is not None:
                self._x_p = np.pad(
                    self._x, (self._window, self._window),
                    'linear_ramp',
                    end_values=(
                        np.min(self._x) - self._window * value_interval[0],
                        np.max(self._x) + self._window * value_interval[0]))

    @property
    def y(self):
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        array_like
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for the **self.y** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

            self._y = value

            if self._window is not None:
                self._y_p = np.pad(self._y, **self._padding_args)

    @property
    def window(self):
        """
        Getter and setter property for the half window.

        Parameters
        ----------
        value : int
            Value to set the half window with.

        Returns
        -------
        int
            Half window.
        """

        return self._window

    @window.setter
    def window(self, value):
        """
        Setter for the **self.window** property.
        """

        if value is not None:
            assert is_integer(value), '"window" must be an integer!'

            assert value >= 1, '"window" must be equal or superior to 1!'

            self._window = value

            if self._x is not None:
                self.x = self._x

            if self._y is not None:
                self.y = self.y

    @property
    def kernel(self):
        """
        Getter and setter property for the kernel callable.

        Parameters
        ----------
        value : callable
            Value to set the kernel callable.

        Returns
        -------
        callable
            Kernel callable.
        """

        return self._kernel

    @kernel.setter
    def kernel(self, value):
        """
        Setter for the **self.kernel** property.
        """

        if value is not None:
            assert hasattr(
                value,
                '__call__'), ('"{0}" attribute: "{1}" is not callable!'.format(
                    'kernel', value))

            self._kernel = value

    @property
    def kernel_args(self):
        """
        Getter and setter property for the kernel call time arguments.

        Parameters
        ----------
        value : dict
            Value to call the interpolation kernel with.

        Returns
        -------
        dict
            Kernel call time arguments.
        """

        return self._kernel_args

    @kernel_args.setter
    def kernel_args(self, value):
        """
        Setter for the **self.kernel_args** property.
        """

        if value is not None:
            assert isinstance(value, (dict, OrderedDict)), (
                '"{0}" attribute: "{1}" type is not "dict" or "OrderedDict"!'
            ).format('kernel_args', value)

            self._kernel_args = value

    @property
    def padding_args(self):
        """
        Getter and setter property for the kernel call time arguments.

        Parameters
        ----------
        value : dict
            Value to call the interpolation kernel with.

        Returns
        -------
        dict
            Kernel call time arguments.
        """

        return self._padding_args

    @padding_args.setter
    def padding_args(self, value):
        """
        Setter for the **self.padding_args** property.
        """

        if value is not None:
            assert isinstance(value, (dict, OrderedDict)), (
                '"{0}" attribute: "{1}" type is not "dict" or "OrderedDict"!'
            ).format('padding_args', value)

            self._padding_args = value

            if self._y is not None:
                self.y = self.y

    def __call__(self, x):
        """
        Evaluates the interpolator at given point(s).

        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_numeric(self._evaluate(x))

        return xi

    def _evaluate(self, x):
        """
        Performs the interpolator evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the interpolant at.

        Returns
        -------
        ndarray
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        x_interval = interval(self._x)[0]
        x_f = np.floor(x / x_interval)

        windows = (x_f[:, np.newaxis] + np.arange(-self._window + 1,
                                                  self._window + 1))
        clip_l = min(self._x_p) / x_interval
        clip_h = max(self._x_p) / x_interval
        windows = np.clip(windows, clip_l, clip_h) - clip_l
        windows = np.around(windows).astype(np.int_)

        return np.sum(
            self._y_p[windows] *
            self._kernel(x[:, np.newaxis] / x_interval - windows -
                         min(self._x_p) / x_interval, **self._kernel_args),
            axis=-1)

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self._x) != len(self._y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(
                     len(self._x), len(self._y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


class LinearInterpolator(object):
    """
    Linearly interpolates a 1-D function.

    Parameters
    ----------
    x : array_like
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : array_like
        Dependent and already known :math:`y` variable values to
        interpolate.
    dtype : type
        Data type used for internal conversions.

    Attributes
    ----------
    x
    y

    Methods
    -------
    __call__

    Notes
    -----
    This class is a wrapper around *numpy.interp* definition.

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = LinearInterpolator(x, y)
    >>> # Doctests ellipsis for Python 2.x compatibility.
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.64...

    Interpolating an *array_like* variable:

    >>> f([0.25, 0.75])
    array([ 6.7825,  8.5075])
    """

    def __init__(self, x, y, dtype=DEFAULT_FLOAT_DTYPE):

        self._x = None
        self._y = None
        self._dtype = dtype

        self.x = x
        self.y = y

        self._validate_dimensions()

    @property
    def x(self):
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        array_like
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value):
        """
        Setter for the **self.x** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

        self._x = value

    @property
    def y(self):
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        array_like
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for the **self.y** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

        self._y = value

    def __call__(self, x):
        """
        Evaluates the interpolating polynomial at given point(s).


        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_numeric(self._evaluate(x))

        return xi

    def _evaluate(self, x):
        """
        Performs the interpolating polynomial evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the interpolant at.

        Returns
        -------
        ndarray
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        return np.interp(x, self._x, self._y)

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self._x) != len(self._y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(
                     len(self._x), len(self._y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


class SpragueInterpolator(object):
    """
    Constructs a fifth-order polynomial that passes through :math:`y` dependent
    variable.

    *Sprague (1880)* method is recommended by the *CIE* for interpolating
    functions having a uniformly spaced independent variable.

    Parameters
    ----------
    x : array_like
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : array_like
        Dependent and already known :math:`y` variable values to
        interpolate.
    dtype : type
        Data type used for internal conversions.

    Attributes
    ----------
    x
    y

    Methods
    -------
    __call__

    Notes
    -----
    The minimum number :math:`k` of data points required along the
    interpolation axis is :math:`k=6`.

    References
    ----------
    .. [3]  CIE TC 1-38. (2005). 9.2.4 Method of interpolation for uniformly
            spaced independent variable. In CIE 167:2005 Recommended Practice
            for Tabulating Spectral Data for Use in Colour Computations
            (pp. 1–27). ISBN:978-3-901-90641-1
    .. [4]  Westland, S., Ripamonti, C., & Cheung, V. (2012). Interpolation
            Methods. In Computational Colour Science Using MATLAB
            (2nd ed., pp. 29–37). ISBN:978-0-470-66569-5

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = SpragueInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.2185025...

    Interpolating an *array_like* variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.7295161...,  7.8140625...])
    """

    SPRAGUE_C_COEFFICIENTS = np.array(
        [[884, -1960, 3033, -2648, 1080, -180],
         [508, -540, 488, -367, 144,
          -24], [-24, 144, -367, 488, -540, 508],
         [-180, 1080, -2648, 3033, -1960, 884]])  # yapf: disable
    """
    Defines the coefficients used to generate extra points for boundaries
    interpolation.

    SPRAGUE_C_COEFFICIENTS : array_like, (4, 6)

    References
    ----------
    .. [5]  CIE TC 1-38. (2005). Table V. Values of the c-coefficients of
            Equ.s 6 and 7. In CIE 167:2005 Recommended Practice for Tabulating
            Spectral Data for Use in Colour Computations (p. 19).
            ISBN:978-3-901-90641-1
    """

    def __init__(self, x, y, dtype=DEFAULT_FLOAT_DTYPE):
        self._xp = None
        self._yp = None

        self._x = None
        self._y = None
        self._dtype = dtype

        self.x = x
        self.y = y

        self._validate_dimensions()

    @property
    def x(self):
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        array_like
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value):
        """
        Setter for the **self.x** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

            value_interval = interval(value)[0]

            xp1 = value[0] - value_interval * 2
            xp2 = value[0] - value_interval
            xp3 = value[-1] + value_interval
            xp4 = value[-1] + value_interval * 2

            self._xp = np.concatenate(((xp1, xp2), value, (xp3, xp4)))

        self._x = value

    @property
    def y(self):
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        array_like
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for the **self.y** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

            assert len(value) >= 6, (
                '"y" dependent variable values count must be in domain [6:]!')

            yp1 = np.ravel((np.dot(self.SPRAGUE_C_COEFFICIENTS[0],
                                   np.array(value[0:6]).reshape(
                                       (6, 1)))) / 209)[0]
            yp2 = np.ravel((np.dot(self.SPRAGUE_C_COEFFICIENTS[1],
                                   np.array(value[0:6]).reshape(
                                       (6, 1)))) / 209)[0]
            yp3 = np.ravel((np.dot(self.SPRAGUE_C_COEFFICIENTS[2],
                                   np.array(value[-6:]).reshape(
                                       (6, 1)))) / 209)[0]
            yp4 = np.ravel((np.dot(self.SPRAGUE_C_COEFFICIENTS[3],
                                   np.array(value[-6:]).reshape(
                                       (6, 1)))) / 209)[0]

            self._yp = np.concatenate(((yp1, yp2), value, (yp3, yp4)))

        self._y = value

    def __call__(self, x):
        """
        Evaluates the interpolating polynomial at given point(s).

        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        numeric or ndarray
            Interpolated value(s).
        """

        return self._evaluate(x)

    def _evaluate(self, x):
        """
        Performs the interpolating polynomial evaluation at given point.

        Parameters
        ----------
        x : numeric
            Point to evaluate the interpolant at.

        Returns
        -------
        float
            Interpolated point values.
        """

        x = np.asarray(x)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        i = np.searchsorted(self._xp, x) - 1
        X = (x - self._xp[i]) / (self._xp[i + 1] - self._xp[i])

        r = self._yp

        a0p = r[i]
        a1p = ((2 * r[i - 2] - 16 * r[i - 1] + 16 * r[i + 1] -
                2 * r[i + 2]) / 24)  # yapf: disable
        a2p = ((-r[i - 2] + 16 * r[i - 1] - 30 * r[i] + 16 * r[i + 1] -
                r[i + 2]) / 24)  # yapf: disable
        a3p = ((-9 * r[i - 2] + 39 * r[i - 1] - 70 * r[i] + 66 * r[i + 1] -
                33 * r[i + 2] + 7 * r[i + 3]) / 24)
        a4p = ((13 * r[i - 2] - 64 * r[i - 1] + 126 * r[i] - 124 * r[i + 1] +
                61 * r[i + 2] - 12 * r[i + 3]) / 24)
        a5p = ((-5 * r[i - 2] + 25 * r[i - 1] - 50 * r[i] + 50 * r[i + 1] -
                25 * r[i + 2] + 5 * r[i + 3]) / 24)

        y = (a0p + a1p * X + a2p * X ** 2 + a3p * X ** 3 + a4p * X ** 4 +
             a5p * X ** 5)

        return y

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self._x) != len(self._y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(
                     len(self._x), len(self._y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


class CubicSplineInterpolator(scipy.interpolate.interp1d):
    """
    Interpolates a 1-D function using cubic spline interpolation.

    Notes
    -----
    This class is a wrapper around *scipy.interpolate.interp1d* class.
    """

    def __init__(self, *args, **kwargs):
        super(CubicSplineInterpolator, self).__init__(
            kind='cubic', *args, **kwargs)


class PchipInterpolator(scipy.interpolate.PchipInterpolator):
    """
    Interpolates a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial interpolation.

    Attributes
    ----------
    y

    Notes
    -----
    -   This class is a wrapper around *scipy.interpolate.PchipInterpolator*
        class.
    """

    def __init__(self, x, y, *args, **kwargs):
        super(PchipInterpolator, self).__init__(x, y, *args, **kwargs)

        self._y = y

    @property
    def y(self):
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        array_like
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for the **self.y** property.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('y'))


class NullInterpolator(object):
    """
    Performs 1-D function null interpolation, i.e. a call within given
    tolerances will return existing :math:`y` variable values and `default` if
    outside tolerances.

    Parameters
    ----------
    x : ndarray
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : ndarray
        Dependent and already known :math:`y` variable values to
        interpolate.
    absolute_tolerance : numeric, optional
        Absolute tolerance.
    relative_tolerance : numeric, optional
        Relative tolerance.
    default : numeric, optional
        Default value for interpolation outside tolerances.
    dtype : type
        Data type used for internal conversions.

    Attributes
    ----------
    x
    y
    relative_tolerance
    absolute_tolerance
    default

    Methods
    -------
    __call__

    Examples
    --------
    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = NullInterpolator(x, y)
    >>> f(0.5)
    nan
    >>> f(1.0)  # doctest: +ELLIPSIS
    9.3699999...
    >>> f = NullInterpolator(x, y, absolute_tolerance=0.01)
    >>> f(1.01)  # doctest: +ELLIPSIS
    9.3699999...
    """

    def __init__(self,
                 x,
                 y,
                 absolute_tolerance=10e-7,
                 relative_tolerance=10e-7,
                 default=np.nan,
                 dtype=DEFAULT_FLOAT_DTYPE):
        self._x = None
        self._y = None
        self._absolute_tolerance = None
        self._relative_tolerance = None
        self._default = None
        self._dtype = dtype

        self.x = x
        self.y = y
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.default = default

        self._validate_dimensions()

    @property
    def x(self):
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        array_like
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value):
        """
        Setter for the **self.x** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

        self._x = value

    @property
    def y(self):
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        array_like
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for the **self.y** property.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(self._dtype)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

        self._y = value

    @property
    def relative_tolerance(self):
        """
        Getter and setter property for the relative tolerance.

        Parameters
        ----------
        value : numeric
            Value to set the relative tolerance with.

        Returns
        -------
        numeric
            Relative tolerance.
        """

        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value):
        """
        Setter for the **self.relative_tolerance** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"relative_tolerance" variable must be a "numeric"!')

        self._relative_tolerance = value

    @property
    def absolute_tolerance(self):
        """
        Getter and setter property for the absolute tolerance.

        Parameters
        ----------
        value : numeric
            Value to set the absolute tolerance with.

        Returns
        -------
        numeric
            Absolute tolerance.
        """

        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value):
        """
        Setter for the **self.absolute_tolerance** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"absolute_tolerance" variable must be a "numeric"!')

        self._absolute_tolerance = value

    @property
    def default(self):
        """
        Getter and setter property for the default value for call outside
        tolerances.

        Parameters
        ----------
        value : numeric
            Value to set the default value with.

        Returns
        -------
        numeric
            Default value.
        """

        return self._default

    @default.setter
    def default(self, value):
        """
        Setter for the **self.default** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"default" variable must be a "numeric"!')

        self._default = value

    def __call__(self, x):
        """
        Evaluates the interpolator at given point(s).


        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(self._dtype)

        xi = as_numeric(self._evaluate(x))

        return xi

    def _evaluate(self, x):
        """
        Performs the interpolator evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the interpolant at.

        Returns
        -------
        ndarray
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        indexes = closest_indexes(self._x, x)
        values = self._y[indexes]
        values[~np.isclose(
            self._x[indexes],
            x,
            rtol=self._absolute_tolerance,
            atol=self._relative_tolerance)] = self._default

        return values

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self._x) != len(self._y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(
                     len(self._x), len(self._y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


def lagrange_coefficients(r, n=4):
    """
    Computes the *Lagrange Coefficients* at given point :math:`r` for degree
    :math:`n`.

    Parameters
    ----------
    r : numeric
        Point to get the *Lagrange Coefficients* at.
    n : int, optional
        Degree of the *Lagrange Coefficients* being calculated.

    Returns
    -------
    ndarray

    References
    ----------
    .. [4]  Fairman, H. S. (1985). The calculation of weight factors for
            tristimulus integration. Color Research & Application, 10(4),
            199–203. doi:10.1002/col.5080100407
    .. [5]  Wikipedia. (n.d.). Lagrange polynomial - Definition. Retrieved
            January 20, 2016, from
            https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition

    Examples
    --------
    >>> lagrange_coefficients(0.1)
    array([ 0.8265,  0.2755, -0.1305,  0.0285])
    """

    r_i = np.arange(n)
    L_n = []
    for j in range(len(r_i)):
        basis = [(r - r_i[i]) / (r_i[j] - r_i[i]) for i in range(len(r_i))
                 if i != j]
        L_n.append(reduce(lambda x, y: x * y, basis))  # noqa

    return np.array(L_n)
