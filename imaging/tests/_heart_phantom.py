"""Synthetic heart phantom (copy of the CardioSolv extension test fixture)."""

import numpy as np


def synthetic_heart_masks(h=1.0):
    ax = np.arange(-70, 100 + h, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    base = 15.0
    epi = (X**2 + Y**2) / 32**2 + (Z - base) ** 2 / 70**2 <= 1
    endo = (X**2 + Y**2) / 21**2 + (Z - base) ** 2 / 59**2 <= 1
    below = Z <= base
    myo = epi & ~endo & below
    lv = endo & below
    epi_pad = (X**2 + Y**2) / 32.5**2 + (Z - base) ** 2 / 70.5**2 <= 1
    rv = ((X + 30) ** 2 / 26**2 + Y**2 / 34**2 + (Z - 12) ** 2 / 55**2 <= 1) & (Z <= 12) & ~epi_pad
    la = (X - 6) ** 2 + Y**2 + (Z - 33) ** 2 <= 18**2
    ra = (X + 34) ** 2 + Y**2 + (Z - 31) ** 2 <= 15**2
    ao = ((X - 12) ** 2 + (Y - 14) ** 2 <= 11**2) & (Z >= 8) & (Z <= 90) & ~la
    origin = np.array([ax[0]] * 3)
    return {"myo": myo, "lv": lv, "rv": rv, "la": la, "ra": ra, "ao": ao}, origin, h
