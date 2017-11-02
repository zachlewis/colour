#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.xrite` module.
"""

from __future__ import division, unicode_literals

import os
import unittest

from colour.colorimetry import SpectralPowerDistribution
from colour.io import read_spds_from_xrite_file

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'COLOURCHECKER_XRITE_1', 'TestReadSpdsFromXRiteFile'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

COLOURCHECKER_XRITE_1 = {
    380.0: 0.0069,
    390.0: 0.0069,
    400.0: 0.0068,
    410.0: 0.0068,
    420.0: 0.0073,
    430.0: 0.0075,
    440.0: 0.0065,
    450.0: 0.0074,
    460.0: 0.0073,
    470.0: 0.0073,
    480.0: 0.0074,
    490.0: 0.0074,
    500.0: 0.0075,
    510.0: 0.0075,
    520.0: 0.0072,
    530.0: 0.0072,
    540.0: 0.0072,
    550.0: 0.0072,
    560.0: 0.0072,
    570.0: 0.0071,
    580.0: 0.0071,
    590.0: 0.0071,
    600.0: 0.0071,
    610.0: 0.0072,
    620.0: 0.0071,
    630.0: 0.0071,
    640.0: 0.0071,
    650.0: 0.0070,
    660.0: 0.0074,
    670.0: 0.0068,
    680.0: 0.0067,
    690.0: 0.0067,
    700.0: 0.0066,
    710.0: 0.0066,
    720.0: 0.0066,
    730.0: 0.0065
}


class TestReadSpdsFromXRiteFile(unittest.TestCase):
    """
    Defines :func:`colour.io.xrite.read_spds_from_xrite_file` definition units
    tests methods.
    """

    def test_read_spds_from_xrite_file(self):
        """
        Tests :func:`colour.io.xrite.read_spds_from_xrite_file` definition.
        """

        colour_checker_xrite = os.path.join(RESOURCES_DIRECTORY,
                                            'xrite_digital_colour_checker.txt')
        spds = read_spds_from_xrite_file(colour_checker_xrite)
        for spd in spds.values():
            self.assertIsInstance(spd, SpectralPowerDistribution)

        self.assertEqual(spds['X1'],
                         SpectralPowerDistribution(
                             COLOURCHECKER_XRITE_1, name='X1'))


if __name__ == '__main__':
    unittest.main()
