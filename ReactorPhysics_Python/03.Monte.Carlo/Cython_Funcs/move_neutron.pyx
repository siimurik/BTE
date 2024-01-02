import numpy as np
cimport numpy as np
cimport cython

#
def move_neutron(double[:] x, double[:] y, int iNeutron, double pitch, double freePath, double dirX, double dirY):
    x[iNeutron] += freePath * dirX
    y[iNeutron] += freePath * dirY

    # If outside the cell, find the corresponding point inside the cell
    while x[iNeutron] < 0:
        x[iNeutron] += pitch
    while y[iNeutron] < 0:
        y[iNeutron] += pitch
    while x[iNeutron] > pitch:
        x[iNeutron] -= pitch
    while y[iNeutron] > pitch:
        y[iNeutron] -= pitch

    return x, y