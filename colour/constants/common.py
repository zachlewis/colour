#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Constants
================

Defines common constants objects that don't belong to any specific category.
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'FLOATING_POINT_NUMBER_PATTERN', 'INTEGER_THRESHOLD', 'EPSILON',
    'DEFAULT_FLOAT_DTYPE'
]

FLOATING_POINT_NUMBER_PATTERN = '[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
"""
Floating point number regex matching pattern.

FLOATING_POINT_NUMBER_PATTERN : unicode
"""

INTEGER_THRESHOLD = 1e-3
"""
Integer threshold value.

INTEGER_THRESHOLD : numeric
"""

EPSILON = np.finfo(np.float_).eps
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.

EPSILON : numeric
"""

DEFAULT_FLOAT_DTYPE = np.float_
"""
Default floating point number dtype.

DEFAULT_FLOAT_DTYPE : type
"""
