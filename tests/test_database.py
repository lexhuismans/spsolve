import math

import numpy as np
import pytest
from spsolve import database as db


def test_get_m_e():
    material1 = 'GaAs'
    material2 = 'AlGaAs'
    x = .3
    assert db.get_m_e(material1) == 0.067
    assert db.get_m_e(material2, x) == 0.0929


def test_get_band_gap():
    material1 = 'GaAs'
    material2 = 'AlGaAs'
    x = .3
    assert db.get_band_gap(material1) == 1.4223
    assert db.get_band_gap(material2, x) == 1.6542


def test_get_dielectric_constant():
    material1 = 'GaAs'
    material2 = 'AlGaAs'
    x = .3
    assert db.get_dielectric_constant(material1) == 12.9
    assert db.get_dielectric_constant(material2, x) == 12.048


def test_get_band_offset():
    material1 = 'GaAs'
    material2 = 'AlGaAs'
    x = .3
    CBO1 = db.get_band_offset(material1)
    CBO2 = db.get_band_offset(material2, x)
    assert CBO2-CBO1 == pytest.approx(.2319, .01)
