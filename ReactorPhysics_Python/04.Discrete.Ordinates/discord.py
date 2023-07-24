import os
import h5py
import time as t
import numpy as np
import numba as nb
import scipy as sp
import scipy.sparse.linalg as spla
from pyXSteam.XSteam import XSteam

def getLebedevReccurencePoints(type, start, a, b, v, leb):
    c = 0.0
    np.pi = np.pi

    if type == 1:
        a = 1.0
        leb["x"][start] = a
        leb["y"][start] = 0.0
        leb["z"][start] = 0.0
        leb["w"][start] = 4.0 * np.pi * v
        leb["x"][start+1] = -a
        leb["y"][start+1] = 0.0
        leb["z"][start+1] = 0.0
        leb["w"][start+1] = 4.0 * np.pi * v
        leb["x"][start+2] = 0.0
        leb["y"][start+2] = a
        leb["z"][start+2] = 0.0
        leb["w"][start+2] = 4.0 * np.pi * v
        leb["x"][start+3] = 0.0
        leb["y"][start+3] = -a
        leb["z"][start+3] = 0.0
        leb["w"][start+3] = 4.0 * np.pi * v
        leb["x"][start+4] = 0.0
        leb["y"][start+4] = 0.0
        leb["z"][start+4] = a
        leb["w"][start+4] = 4.0 * np.pi * v
        leb["x"][start+5] = 0.0
        leb["y"][start+5] = 0.0
        leb["z"][start+5] = -a
        leb["w"][start+5] = 4.0 * np.pi * v
        start = start + 6

    elif type == 2:
        a = np.sqrt(0.5)
        leb["x"][start] = 0.0
        leb["y"][start] = a
        leb["z"][start] = a
        leb["w"][start] = 4.0 * np.pi * v
        leb["x"][start+1] = 0.0
        leb["y"][start+1] = -a
        leb["z"][start+1] = a
        leb["w"][start+1] = 4.0 * np.pi * v
        leb["x"][start+2] = 0.0
        leb["y"][start+2] = a
        leb["z"][start+2] = -a
        leb["w"][start+2] = 4.0 * np.pi * v
        leb["x"][start+3] = 0.0
        leb["y"][start+3] = -a
        leb["z"][start+3] = -a
        leb["w"][start+3] = 4.0 * np.pi * v
        leb["x"][start+4] = a
        leb["y"][start+4] = 0.0
        leb["z"][start+4] = a
        leb["w"][start+4] = 4.0 * np.pi * v
        leb["x"][start+5] = a
        leb["y"][start+5] = 0.0
        leb["z"][start+5] = -a
        leb["w"][start+5] = 4.0 * np.pi * v
        leb["x"][start+6] = -a
        leb["y"][start+6] = 0.0
        leb["z"][start+6] = a
        leb["w"][start+6] = 4.0 * np.pi * v
        leb["x"][start+7] = -a
        leb["y"][start+7] = 0.0
        leb["z"][start+7] = -a
        leb["w"][start+7] = 4.0 * np.pi * v
        leb["x"][start+8] = a
        leb["y"][start+8] = a
        leb["z"][start+8] = 0.0
        leb["w"][start+8] = 4.0 * np.pi * v
        leb["x"][start+9] = -a
        leb["y"][start+9] = a
        leb["z"][start+9] = 0.0
        leb["w"][start+9] = 4.0 * np.pi * v
        leb["x"][start+10] = a
        leb["y"][start+10] = -a
        leb["z"][start+10] = 0.0
        leb["w"][start+10] = 4.0 * np.pi * v
        leb["x"][start+11] = -a
        leb["y"][start+11] = -a
        leb["z"][start+11] = 0.0
        leb["w"][start+11] = 4.0 * np.pi * v
        start = start + 12

    elif type == 3:
        a = np.sqrt(1.0 / 3.0)
        leb["x"][start] = a
        leb["y"][start] = a
        leb["z"][start] = a
        leb["w"][start] = 4.0 * np.pi * v
        leb["x"][start+1] = -a
        leb["y"][start+1] = a
        leb["z"][start+1] = a
        leb["w"][start+1] = 4.0 * np.pi * v
        leb["x"][start+2] = a
        leb["y"][start+2] = -a
        leb["z"][start+2] = a
        leb["w"][start+2] = 4.0 * np.pi * v
        leb["x"][start+3] = a
        leb["y"][start+3] = a
        leb["z"][start+3] = -a
        leb["w"][start+3] = 4.0 * np.pi * v
        leb["x"][start+4] = -a
        leb["y"][start+4] = -a
        leb["z"][start+4] = a
        leb["w"][start+4] = 4.0 * np.pi * v
        leb["x"][start+5] = a
        leb["y"][start+5] = -a
        leb["z"][start+5] = -a
        leb["w"][start+5] = 4.0 * np.pi * v
        leb["x"][start+6] = -a
        leb["y"][start+6] = a
        leb["z"][start+6] = -a
        leb["w"][start+6] = 4.0 * np.pi * v
        leb["x"][start+7] = -a
        leb["y"][start+7] = -a
        leb["z"][start+7] = -a
        leb["w"][start+7] = 4.0 * np.pi * v
        start = start + 8

    elif type == 4:
        #%/* In this case A is inputed */
        b = np.sqrt(1.0 - 2.0*a*a)
        leb["x"][start] = a
        leb["y"][start] = a
        leb["z"][start] = b
        leb["w"][start] = 4.0*np.pi*v
        leb["x"][start+1] = -a
        leb["y"][start+1] = a
        leb["z"][start+1] = b
        leb["w"][start+1] = 4.0*np.pi*v
        leb["x"][start+2] = a
        leb["y"][start+2] = -a
        leb["z"][start+2] = b
        leb["w"][start+2] = 4.0*np.pi*v
        leb["x"][start+3] = a
        leb["y"][start+3] = a
        leb["z"][start+3] = -b
        leb["w"][start+3] = 4.0*np.pi*v
        leb["x"][start+4] = -a
        leb["y"][start+4] = -a
        leb["z"][start+4] = b
        leb["w"][start+4] = 4.0*np.pi*v
        leb["x"][start+5] = -a
        leb["y"][start+5] = a
        leb["z"][start+5] = -b
        leb["w"][start+5] = 4.0*np.pi*v
        leb["x"][start+6] = a
        leb["y"][start+6] = -a
        leb["z"][start+6] = -b
        leb["w"][start+6] = 4.0*np.pi*v
        leb["x"][start+7] = -a
        leb["y"][start+7] = -a
        leb["z"][start+7] = -b
        leb["w"][start+7] = 4.0*np.pi*v
        leb["x"][start+8] = -a
        leb["y"][start+8] = b
        leb["z"][start+8] = a
        leb["w"][start+8] = 4.0*np.pi*v
        leb["x"][start+9] = a
        leb["y"][start+9] = -b
        leb["z"][start+9] = a
        leb["w"][start+9] = 4.0*np.pi*v
        leb["x"][start+10] = a
        leb["y"][start+10] = b
        leb["z"][start+10] = -a
        leb["w"][start+10] = 4.0*np.pi*v
        leb["x"][start+11] = -a
        leb["y"][start+11] = -b
        leb["z"][start+11] = a
        leb["w"][start+11] = 4.0*np.pi*v
        leb["x"][start+12] = -a
        leb["y"][start+12] = b
        leb["z"][start+12] = -a
        leb["w"][start+12] = 4.0*np.pi*v
        leb["x"][start+13] = a
        leb["y"][start+13] = -b
        leb["z"][start+13] = -a
        leb["w"][start+13] = 4.0*np.pi*v
        leb["x"][start+14] = -a
        leb["y"][start+14] = -b
        leb["z"][start+14] = -a
        leb["w"][start+14] = 4.0*np.pi*v
        leb["x"][start+15] = a
        leb["y"][start+15] = b
        leb["z"][start+15] = a
        leb["w"][start+15] = 4.0*np.pi*v
        leb["x"][start+16] = b
        leb["y"][start+16] = a
        leb["z"][start+16] = a
        leb["w"][start+16] = 4.0*np.pi*v
        leb["x"][start+17] = -b
        leb["y"][start+17] = a
        leb["z"][start+17] = a
        leb["w"][start+17] = 4.0*np.pi*v
        leb["x"][start+18] = b
        leb["y"][start+18] = -a
        leb["z"][start+18] = a
        leb["w"][start+18] = 4.0*np.pi*v
        leb["x"][start+19] = b
        leb["y"][start+19] = a
        leb["z"][start+19] = -a
        leb["w"][start+19] = 4.0*np.pi*v
        leb["x"][start+20] = -b
        leb["y"][start+20] = -a
        leb["z"][start+20] = a
        leb["w"][start+20] = 4.0*np.pi*v
        leb["x"][start+21] = -b
        leb["y"][start+21] = a
        leb["z"][start+21] = -a
        leb["w"][start+21] = 4.0*np.pi*v
        leb["x"][start+22] = b
        leb["y"][start+22] = -a
        leb["z"][start+22] = -a
        leb["w"][start+22] = 4.0*np.pi*v
        leb["x"][start+23] = -b
        leb["y"][start+23] = -a
        leb["z"][start+23] = -a
        leb["w"][start+23] = 4.0*np.pi*v
        start = start + 24
    
    elif type == 5:
        #%/* A is inputed in this case as well*/
        b = np.sqrt(1-a*a)
        leb["x"][start] = a
        leb["y"][start] = b
        leb["z"][start] = 0.0
        leb["w"][start] = 4.0*np.pi*v
        leb["x"][start+1] = -a
        leb["y"][start+1] = b
        leb["z"][start+1] = 0.0
        leb["w"][start+1] = 4.0*np.pi*v
        leb["x"][start+2] = a
        leb["y"][start+2] = -b
        leb["z"][start+2] = 0.0
        leb["w"][start+2] = 4.0*np.pi*v
        leb["x"][start+3] = -a
        leb["y"][start+3] = -b
        leb["z"][start+3] = 0.0
        leb["w"][start+3] = 4.0*np.pi*v
        leb["x"][start+4] = b
        leb["y"][start+4] = a
        leb["z"][start+4] = 0.0
        leb["w"][start+4] = 4.0*np.pi*v
        leb["x"][start+5] = -b
        leb["y"][start+5] = a
        leb["z"][start+5] = 0.0
        leb["w"][start+5] = 4.0*np.pi*v
        leb["x"][start+6] = b
        leb["y"][start+6] = -a
        leb["z"][start+6] = 0.0
        leb["w"][start+6] = 4.0*np.pi*v
        leb["x"][start+7] = -b
        leb["y"][start+7] = -a
        leb["z"][start+7] = 0.0
        leb["w"][start+7] = 4.0*np.pi*v
        leb["x"][start+8] = a
        leb["y"][start+8] = 0.0
        leb["z"][start+8] = b
        leb["w"][start+8] = 4.0*np.pi*v
        leb["x"][start+9] = -a
        leb["y"][start+9] = 0.0
        leb["z"][start+9] = b
        leb["w"][start+9] = 4.0*np.pi*v
        leb["x"][start+10] = a
        leb["y"][start+10] = 0.0
        leb["z"][start+10] = -b
        leb["w"][start+10] = 4.0*np.pi*v
        leb["x"][start+11] = -a
        leb["y"][start+11] = 0.0
        leb["z"][start+11] = -b
        leb["w"][start+11] = 4.0*np.pi*v
        leb["x"][start+12] = b
        leb["y"][start+12] = 0.0
        leb["z"][start+12] = a
        leb["w"][start+12] = 4.0*np.pi*v
        leb["x"][start+13] = -b
        leb["y"][start+13] = 0.0
        leb["z"][start+13] = a
        leb["w"][start+13] = 4.0*np.pi*v
        leb["x"][start+14] = b
        leb["y"][start+14] = 0.0
        leb["z"][start+14] = -a
        leb["w"][start+14] = 4.0*np.pi*v
        leb["x"][start+15] = -b
        leb["y"][start+15] = 0.0
        leb["z"][start+15] = -a
        leb["w"][start+15] = 4.0*np.pi*v
        leb["x"][start+16] = 0.0
        leb["y"][start+16] = a
        leb["z"][start+16] = b
        leb["w"][start+16] = 4.0*np.pi*v
        leb["x"][start+17] = 0.0
        leb["y"][start+17] = -a
        leb["z"][start+17] = b
        leb["w"][start+17] = 4.0*np.pi*v
        leb["x"][start+18] = 0.0
        leb["y"][start+18] = a
        leb["z"][start+18] = -b
        leb["w"][start+18] = 4.0*np.pi*v
        leb["x"][start+19] = 0.0
        leb["y"][start+19] = -a
        leb["z"][start+19] = -b
        leb["w"][start+19] = 4.0*np.pi*v
        leb["x"][start+20] = 0.0
        leb["y"][start+20] = b
        leb["z"][start+20] = a
        leb["w"][start+20] = 4.0*np.pi*v
        leb["x"][start+21] = 0.0
        leb["y"][start+21] = -b
        leb["z"][start+21] = a
        leb["w"][start+21] = 4.0*np.pi*v
        leb["x"][start+22] = 0.0
        leb["y"][start+22] = b
        leb["z"][start+22] = -a
        leb["w"][start+22] = 4.0*np.pi*v
        leb["x"][start+23] = 0.0
        leb["y"][start+23] = -b
        leb["z"][start+23] = -a
        leb["w"][start+23] = 4.0*np.pi*v
        start = start + 24
    
    elif type == 6:
        #%/* both A and B are inputed in this case */
        c = np.sqrt(1.0 - a*a - b*b)
        leb["x"][start] = a
        leb["y"][start] = b
        leb["z"][start] = c
        leb["w"][start] = 4.0*np.pi*v
        leb["x"][start+1] = -a
        leb["y"][start+1] = b
        leb["z"][start+1] = c
        leb["w"][start+1] = 4.0*np.pi*v
        leb["x"][start+2] = a
        leb["y"][start+2] = -b
        leb["z"][start+2] = c
        leb["w"][start+2] = 4.0*np.pi*v
        leb["x"][start+3] = a
        leb["y"][start+3] = b
        leb["z"][start+3] = -c
        leb["w"][start+3] = 4.0*np.pi*v
        leb["x"][start+4] = -a
        leb["y"][start+4] = -b
        leb["z"][start+4] = c
        leb["w"][start+4] = 4.0*np.pi*v
        leb["x"][start+5] = a
        leb["y"][start+5] = -b
        leb["z"][start+5] = -c
        leb["w"][start+5] = 4.0*np.pi*v
        leb["x"][start+6] = -a
        leb["y"][start+6] = b
        leb["z"][start+6] = -c
        leb["w"][start+6] = 4.0*np.pi*v
        leb["x"][start+7] = -a
        leb["y"][start+7] = -b
        leb["z"][start+7] = -c
        leb["w"][start+7] = 4.0*np.pi*v
        leb["x"][start+8] = b
        leb["y"][start+8] = a
        leb["z"][start+8] = c
        leb["w"][start+8] = 4.0*np.pi*v
        leb["x"][start+9] = -b
        leb["y"][start+9] = a
        leb["z"][start+9] = c
        leb["w"][start+9] = 4.0*np.pi*v
        leb["x"][start+10] = b
        leb["y"][start+10] = -a
        leb["z"][start+10] = c
        leb["w"][start+10] = 4.0*np.pi*v
        leb["x"][start+11] = b
        leb["y"][start+11] = a
        leb["z"][start+11] = -c
        leb["w"][start+11] = 4.0*np.pi*v
        leb["x"][start+12] = -b
        leb["y"][start+12] = -a
        leb["z"][start+12] = c
        leb["w"][start+12] = 4.0*np.pi*v
        leb["x"][start+13] = b
        leb["y"][start+13] = -a
        leb["z"][start+13] = -c
        leb["w"][start+13] = 4.0*np.pi*v
        leb["x"][start+14] = -b
        leb["y"][start+14] = a
        leb["z"][start+14] = -c
        leb["w"][start+14] = 4.0*np.pi*v
        leb["x"][start+15] = -b
        leb["y"][start+15] = -a
        leb["z"][start+15] = -c
        leb["w"][start+15] = 4.0*np.pi*v
        leb["x"][start+16] = c
        leb["y"][start+16] = a
        leb["z"][start+16] = b
        leb["w"][start+16] = 4.0*np.pi*v
        leb["x"][start+17] = -c
        leb["y"][start+17] = a
        leb["z"][start+17] = b
        leb["w"][start+17] = 4.0*np.pi*v
        leb["x"][start+18] = c
        leb["y"][start+18] = -a
        leb["z"][start+18] = b
        leb["w"][start+18] = 4.0*np.pi*v
        leb["x"][start+19] = c
        leb["y"][start+19] = a
        leb["z"][start+19] = -b
        leb["w"][start+19] = 4.0*np.pi*v
        leb["x"][start+20] = -c
        leb["y"][start+20] = -a
        leb["z"][start+20] = b
        leb["w"][start+20] = 4.0*np.pi*v
        leb["x"][start+21] = c
        leb["y"][start+21] = -a
        leb["z"][start+21] = -b
        leb["w"][start+21] = 4.0*np.pi*v
        leb["x"][start+22] = -c
        leb["y"][start+22] = a
        leb["z"][start+22] = -b
        leb["w"][start+22] = 4.0*np.pi*v
        leb["x"][start+23] = -c
        leb["y"][start+23] = -a
        leb["z"][start+23] = -b
        leb["w"][start+23] = 4.0*np.pi*v
        leb["x"][start+24] = c
        leb["y"][start+24] = b
        leb["z"][start+24] = a
        leb["w"][start+24] = 4.0*np.pi*v
        leb["x"][start+25] = -c
        leb["y"][start+25] = b
        leb["z"][start+25] = a
        leb["w"][start+25] = 4.0*np.pi*v
        leb["x"][start+26] = c
        leb["y"][start+26] = -b
        leb["z"][start+26] = a
        leb["w"][start+26] = 4.0*np.pi*v
        leb["x"][start+27] = c
        leb["y"][start+27] = b
        leb["z"][start+27] = -a
        leb["w"][start+27] = 4.0*np.pi*v
        leb["x"][start+28] = -c
        leb["y"][start+28] = -b
        leb["z"][start+28] = a
        leb["w"][start+28] = 4.0*np.pi*v
        leb["x"][start+29] = c
        leb["y"][start+29] = -b
        leb["z"][start+29] = -a
        leb["w"][start+29] = 4.0*np.pi*v
        leb["x"][start+30] = -c
        leb["y"][start+30] = b
        leb["z"][start+30] = -a
        leb["w"][start+30] = 4.0*np.pi*v
        leb["x"][start+31] = -c
        leb["y"][start+31] = -b
        leb["z"][start+31] = -a
        leb["w"][start+31] = 4.0*np.pi*v
        leb["x"][start+32] = a
        leb["y"][start+32] = c
        leb["z"][start+32] = b
        leb["w"][start+32] = 4.0*np.pi*v
        leb["x"][start+33] = -a
        leb["y"][start+33] = c
        leb["z"][start+33] = b
        leb["w"][start+33] = 4.0*np.pi*v
        leb["x"][start+34] = a
        leb["y"][start+34] = -c
        leb["z"][start+34] = b
        leb["w"][start+34] = 4.0*np.pi*v
        leb["x"][start+35] = a
        leb["y"][start+35] = c
        leb["z"][start+35] = -b
        leb["w"][start+35] = 4.0*np.pi*v
        leb["x"][start+36] = -a
        leb["y"][start+36] = -c
        leb["z"][start+36] = b
        leb["w"][start+36] = 4.0*np.pi*v
        leb["x"][start+37] = a
        leb["y"][start+37] = -c
        leb["z"][start+37] = -b
        leb["w"][start+37] = 4.0*np.pi*v
        leb["x"][start+38] = -a
        leb["y"][start+38] = c
        leb["z"][start+38] = -b
        leb["w"][start+38] = 4.0*np.pi*v
        leb["x"][start+39] = -a
        leb["y"][start+39] = -c
        leb["z"][start+39] = -b
        leb["w"][start+39] = 4.0*np.pi*v
        leb["x"][start+40] = b
        leb["y"][start+40] = c
        leb["z"][start+40] = a
        leb["w"][start+40] = 4.0*np.pi*v
        leb["x"][start+41] = -b
        leb["y"][start+41] = c
        leb["z"][start+41] = a
        leb["w"][start+41] = 4.0*np.pi*v
        leb["x"][start+42] = b
        leb["y"][start+42] = -c
        leb["z"][start+42] = a
        leb["w"][start+42] = 4.0*np.pi*v
        leb["x"][start+43] = b
        leb["y"][start+43] = c
        leb["z"][start+43] = -a
        leb["w"][start+43] = 4.0*np.pi*v
        leb["x"][start+44] = -b
        leb["y"][start+44] = -c
        leb["z"][start+44] = a
        leb["w"][start+44] = 4.0*np.pi*v
        leb["x"][start+45] = b
        leb["y"][start+45] = -c
        leb["z"][start+45] = -a
        leb["w"][start+45] = 4.0*np.pi*v
        leb["x"][start+46] = -b
        leb["y"][start+46] = c
        leb["z"][start+46] = -a
        leb["w"][start+46] = 4.0*np.pi*v
        leb["x"][start+47] = -b
        leb["y"][start+47] = -c
        leb["z"][start+47] = -a
        leb["w"][start+47] = 4.0*np.pi*v
        start = start + 48

    else:
        raise ValueError('Bad grid order')

    return leb, start

def getLebedevSphere(degree):
    # Function implementation (as shown in the previous response)

    leb_tmp = {
        'x': np.zeros(degree),
        'y': np.zeros(degree),
        'z': np.zeros(degree),
        'w': np.zeros(degree),
        'n': degree
    }

    start = 0
    a = 0.0
    b = 0.0
    c = 0.0
    v = 0.0

    if degree == 6:
        v = 0.6666666666666667E-1
        leb_tmp, start = getLebedevReccurencePoints(1, start, a, b, v, leb_tmp)

    elif degree == 14:
        v = 0.6666666666666667E-1
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.7500000000000000E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)

    elif degree == 26:
        v = 0.4761904761904762E-1
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.3809523809523810E-1
        leb_tmp, start = getLebedevReccurencePoints(2,start,a,b,v,leb_tmp)
        v = 0.3214285714285714E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)

    elif degree == 38:
        v = 0.9523809523809524E-2
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.3214285714285714E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)
        a = 0.4597008433809831E+0
        v = 0.2857142857142857E-1
        leb_tmp, start = getLebedevReccurencePoints(5,start,a,b,v,leb_tmp)

    elif degree == 50:
        v = 0.1269841269841270E-1
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.2257495590828924E-1
        leb_tmp, start = getLebedevReccurencePoints(2,start,a,b,v,leb_tmp)
        v = 0.2109375000000000E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)
        a = 0.3015113445777636E+0
        v = 0.2017333553791887E-1
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)

    elif degree == 74:
        v = 0.5130671797338464E-3
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.1660406956574204E-1
        leb_tmp, start = getLebedevReccurencePoints(2,start,a,b,v,leb_tmp)
        v = -0.2958603896103896E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)
        a = 0.4803844614152614E+0
        v = 0.2657620708215946E-1
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.3207726489807764E+0
        v = 0.1652217099371571E-1
        leb_tmp, start = getLebedevReccurencePoints(5,start,a,b,v,leb_tmp)

    elif degree == 86:
        v = 0.1154401154401154E-1
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.1194390908585628E-1
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)
        a = 0.3696028464541502E+0
        v = 0.1111055571060340E-1
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.6943540066026664E+0
        v = 0.1187650129453714E-1
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.3742430390903412E+0
        v = 0.1181230374690448E-1
        leb_tmp, start = getLebedevReccurencePoints(5,start,a,b,v,leb_tmp)

    elif degree == 110:
        v = 0.3828270494937162E-2
        leb_tmp, start = getLebedevReccurencePoints(1,start,a,b,v,leb_tmp)
        v = 0.9793737512487512E-2
        leb_tmp, start = getLebedevReccurencePoints(3,start,a,b,v,leb_tmp)
        a = 0.1851156353447362E+0
        v = 0.8211737283191111E-2
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.6904210483822922E+0
        v = 0.9942814891178103E-2
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.3956894730559419E+0
        v = 0.9595471336070963E-2
        leb_tmp, start = getLebedevReccurencePoints(4,start,a,b,v,leb_tmp)
        a = 0.4783690288121502E+0
        v = 0.9694996361663028E-2
        leb_tmp, start = getLebedevReccurencePoints(5,start,a,b,v,leb_tmp)
    else:
         raise ValueError('Angular grid unrecognized, choices are 6, 14, 26, 38, 50, 74, 86, 110')
        
    return leb_tmp


def initPWR_like():
    #global global_g
    with h5py.File("..//00.Lib/initPWR_like.h5", "w") as hdf:
        g  = hdf.create_group("g")
        th = hdf.create_group("th")
        fr = hdf.create_group("fr")

        #group.attrs["nz"] = 10
        #global_g = group

        # Input fuel rod geometry and nodalization
        g_nz = 10  # number of axial nodes
        g.create_dataset("nz", data=g_nz)

        g_fuel_rIn = 0  # inner fuel radius (m)
        g_fuel_rOut = 4.12e-3  # outer fuel radius (m)
        g_fuel_nr = 20  # number of radial nodes in fuel
        g_fuel = g.create_group("fuel")
        g_fuel.create_dataset("rIn",  data=g_fuel_rIn)
        g_fuel.create_dataset("rOut", data=g_fuel_rOut)
        g_fuel.create_dataset("nr",   data=g_fuel_nr)

        g_clad_rIn = 4.22e-3  # inner clad radius (m)
        g_clad_rOut = 4.75e-3  # outer clad radius (m)
        g_clad_nr = 5
        g_clad = g.create_group("clad")
        g_clad.create_dataset("rIn",  data=g_clad_rIn)
        g_clad.create_dataset("rOut", data=g_clad_rOut)
        g_clad.create_dataset("nr",   data=g_clad_nr)

        g_cool_pitch = 13.3e-3  # square unit cell pitch (m)
        g_cool_rOut = np.sqrt(g_cool_pitch**2 / np.pi)  # equivalent radius of the unit cell (m)
        g_cool = g.create_group("cool")
        g_cool.create_dataset("pitch", data=g_cool_pitch)
        g_cool.create_dataset("rOut",  data=g_cool_rOut)

        g_dz0 = 0.3 * np.ones(g_nz)  # height of the node (m)
        g_dzGasPlenum = 0.2  # height of the fuel rod gas plenum assuming it is empty (m)
        g.create_dataset("dz0", data = g_dz0)
        g.create_dataset("dzGasPlenum", data = g_dzGasPlenum)

        # Input average power rating in fuel
        th_qLHGR0 = np.array([  [0, 10, 1e20],  # time (s)
                                [200e2, 200e2, 200e2]   ])  # linear heat generation rate (W/m)
        th.create_dataset("qLHGR0", data = th_qLHGR0)

        # Input fuel rod parameters
        fr_clad_fastFlux = np.array([   [0, 10, 1e20],  # time (s)
                                        [1e13, 1e13, 1e13]  ])  # fast flux in cladding (1/cm2-s)
        fr_clad = fr.create_group("clad")
        fr_clad.create_dataset("fastFlux", data = fr_clad_fastFlux)

        fr_fuel_FGR = np.array([[0, 10, 1e20],  # time (s)
                                [0.03, 0.03, 0.03]])  # fission gas release (-)
        fr_fuel = fr.create_group("fuel")
        fr_fuel.create_dataset("FGR", data = fr_fuel_FGR)

        fr_ingas_Tplenum = 533  # fuel rod gas plenum temperature (K)
        fr_ingas_p0 = 1  # as-fabricated helium pressure inside fuel rod (MPa)
        fr_fuel_por = 0.05 * np.ones((g_nz, g_fuel_nr))  # initial fuel porosity (-)
        fr_ingas = fr.create_group("ingas")
        fr_ingas.create_dataset("Tplenum", data = fr_ingas_Tplenum)
        fr_ingas.create_dataset("p0",      data = fr_ingas_p0)
        fr_fuel.create_dataset("por",      data = fr_fuel_por)

        # Input channel geometry
        g_aFlow = 8.914e-5 * np.ones(g_nz)  # flow area (m2)
        g.create_dataset("aFlow", data = g_aFlow)

        # Input channel parameters
        th_mdot0_ = np.array([  [0, 10, 1000],  # time (s)
                                [0.3, 0.3, 0.3]])  # flowrate (kg/s) 0.3
        th_p0 = 16  # coolant pressure (MPa)
        th_T0 = 533.0  # inlet temperature (K)
        th.create_dataset("mdot0", data = th_mdot0_)
        th.create_dataset("p0",    data = th_p0)
        th.create_dataset("T0",    data = th_T0)

        # Initialize fuel geometry
        g_fuel_dr0 = (g_fuel_rOut - g_fuel_rIn) / (g_fuel_nr - 1)  # fuel node radial thickness (m)
        g_fuel_r0 = np.arange(g_fuel_rIn, g_fuel_rOut + g_fuel_dr0, g_fuel_dr0)  # fuel node radius (m)
        g_fuel_r0_ = np.concatenate(([g_fuel_rIn], np.interp(np.arange(1.5, g_fuel_nr + 0.5), np.arange(1, g_fuel_nr + 1),
                                                                    g_fuel_r0), [g_fuel_rOut]))  # fuel node boundary (m)
        g_fuel_a0_ = np.transpose(np.tile(2*np.pi*g_fuel_r0_[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_fuel_nr + 1))  # XS area of fuel node boundary (m2)
        g_fuel_v0 = np.transpose(np.tile(np.pi*np.diff(g_fuel_r0_**2)[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_fuel_nr))  # fuel node volume (m3)
        g_fuel_vFrac = (g_fuel_rOut**2 - g_fuel_rIn**2) / g_cool_rOut**2
        g_fuel.create_dataset("dr0",   data = g_fuel_dr0)
        g_fuel.create_dataset("r0",    data = g_fuel_r0)
        g_fuel.create_dataset("r0_",   data = g_fuel_r0_)
        g_fuel.create_dataset("a0_",   data = g_fuel_a0_)
        g_fuel.create_dataset("v0",    data = g_fuel_v0)
        g_fuel.create_dataset("vFrac", data = g_fuel_vFrac)

        # Initialize clad geometry
        g_clad_dr0 = (g_clad_rOut - g_clad_rIn) / (g_clad_nr - 1)  # clad node radial thickness (m)
        g_clad_r0 = np.arange(g_clad_rIn, g_clad_rOut + g_clad_dr0, g_clad_dr0)  # clad node radius (m)
        g_clad_r0_ = np.concatenate(([g_clad_rIn], np.interp(np.arange(1.5, g_clad_nr + 0.5), np.arange(1, g_clad_nr + 1), 
                                                                    g_clad_r0), [g_clad_rOut]))  # clad node boundary (m)
        g_clad_a0_ = np.transpose(np.tile(2 * np.pi * g_clad_r0_[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_clad_nr + 1))  # XS area of clad node boundary (m2)
        g_clad_v0 = np.transpose(np.tile(np.pi*np.diff(g_clad_r0_**2)[:, None], g_nz)) * np.tile(g_dz0[:, None], (1, g_clad_nr))  # clad node volume (m3)
        g_clad_vFrac = (g_clad_rOut**2 - g_clad_rIn**2) / g_cool_rOut**2
        g_clad.create_dataset("dr0",   data = g_clad_dr0)
        g_clad.create_dataset("r0",    data = g_clad_r0)
        g_clad.create_dataset("r0_",   data = g_clad_r0_)
        g_clad.create_dataset("a0_",   data = g_clad_a0_)
        g_clad.create_dataset("v0",    data = g_clad_v0)
        g_clad.create_dataset("vFrac", data = g_clad_vFrac)

        # Initialize gap geometry
        dimensions = tuple(range(1, g_nz+1))
        g_gap_dr0 = (g_clad_rIn - g_fuel_rOut) * np.ones(dimensions)   # initial cold gap (m)
        g_gap_r0_ = (g_clad_rIn + g_fuel_rOut) / 2  # average gap radius (m)
        g_gap_a0_ = (2 * np.pi * g_gap_r0_ * np.ones((g_nz, 1))) * g_dz0  # XS area of the mid-gap (m2)
        g_gap_vFrac = (g_clad_rIn**2 - g_fuel_rOut**2) / g_cool_rOut**2
        g_gap = g.create_group("gap")
        g_gap.create_dataset("dr0",   data = g_gap_dr0.flatten())
        g_gap.create_dataset("r0_",   data = g_gap_r0_)
        g_gap.create_dataset("a0_",   data = g_gap_a0_)
        g_gap.create_dataset("vFrac", data = g_gap_vFrac)

        # Initialize as-fabricated inner volumes and gas amount
        g_vGasPlenum = g_dzGasPlenum * np.pi * g_clad_rIn**2  # gas plenum volume (m3)
        g_vGasGap = g_dz0 * np.pi * (g_clad_rIn**2 - g_fuel_rOut**2)  # gas gap volume (m3)
        g_vGasCentralVoid = g_dz0 * np.pi * g_fuel_rIn**2  # gas central void volume (m3)
        fr_ingas_muHe0 = fr_ingas_p0 * (g_vGasPlenum + np.sum(g_vGasGap + g_vGasCentralVoid)) / (8.31e-6 * 293)  # as-fabricated gas amount inside fuel rod (mole)
        g.create_dataset("vGasPlenum",      data = g_vGasPlenum)
        g.create_dataset("vGasGap",         data = g_vGasGap)
        g.create_dataset("vGasCentralVoid", data = g_vGasCentralVoid)
        fr_ingas.create_dataset("muHe0",    data = fr_ingas_muHe0)

        # Initialize gas gap status
        g_gap_open = np.ones(g_nz)
        g_gap_clsd = np.zeros(g_nz)
        g_gap.create_dataset("open", data = g_gap_open)
        g_gap.create_dataset("clsd", data = g_gap_clsd)

        # Initialize fuel and clad total deformation components
        fr_fuel_eps0 = np.zeros((3, g_nz, g_fuel_nr))
        fr_clad_eps0 = np.zeros((3, g_nz, g_clad_nr))
        fuel_eps0 = fr_fuel.create_group("eps0")
        clad_eps0 = fr_clad.create_group("eps0")
        for i in range(3):
            fr_fuel_eps0[i] = np.zeros((g_nz, g_fuel_nr))
            fr_clad_eps0[i] = np.zeros((g_nz, g_clad_nr))
            fuel_eps0.create_dataset(f"eps0(0,{i})", data = fr_fuel_eps0[i])
            clad_eps0.create_dataset(f"eps0(0,{i})", data = fr_clad_eps0[i])

        # Initialize flow channel geometry
        g_volFlow = g_aFlow * g_dz0  # volume of node (m3)
        g_areaHX = 2 * np.pi * g_clad_rOut * g_dz0  # heat exchange area(m2)(m2)
        g_dHyd = 4 * g_volFlow / g_areaHX  # hydraulic diameter (m)
        g_cool_vFrac = (g_cool_rOut**2 - g_clad_rOut**2) / g_cool_rOut**2
        g.create_dataset("volFlow",    data = g_volFlow)
        g.create_dataset("areaHX",     data = g_areaHX)
        g.create_dataset("dHyd",       data = g_dHyd)
        g_cool.create_dataset("vFrac", data = g_cool_vFrac)

        # Initialize thermal hydraulic parameters
        # Path to steam-water properties
        steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
        th_h0 = steamTable.h_pt(th_p0 / 10, th_T0 - 273) * 1e3  # water enthalpy at core inlet (J/kg)
        th_h = np.ones(g_nz) * th_h0  # initial enthalpy in nodes (kJ/kg)
        th_p = np.ones(g_nz) * th_p0  # initial pressure in nodes (MPa)
        th.create_dataset("h0", data = th_h0)
        th.create_dataset("h", data = th_h)
        th.create_dataset("p", data = th_p)

def readPWR_like(input_keyword):
    # Define the mapping of input keyword to group name
    group_mapping = {
        "fr": "fr",
        "g": "g",
        "th": "th"
    }
    # Define the dictionary to store the datasets
    data = {}
    # Open the HDF5 file
    with h5py.File("..//00.Lib/initPWR_like.h5", "r") as file:
        # Get the group name based on the input keyword
        group_name = group_mapping.get(input_keyword)
        if group_name is not None:
            # Get the group
            group = file[group_name]
            # Iterate over the dataset names in the group
            for dataset_name in group.keys():
                # Read the dataset
                dataset = group[dataset_name]
                # Check if the dataset is a struct
                if isinstance(dataset, h5py.Group):
                    # Create a dictionary to store the struct fields
                    struct_data = {}
                    # Iterate over the fields in the struct
                    for field_name in dataset.keys():
                        # Read the field dataset
                        field_dataset = np.array(dataset[field_name])
                        # Store the field dataset in the struct dictionary
                        struct_data[field_name] = field_dataset
                    # Store the struct data in the main data dictionary
                    data[dataset_name] = struct_data
                else:
                    # Read the dataset as a regular array
                    dataset_array = np.array(dataset)
                    # Store the dataset array in the dictionary
                    data[dataset_name] = dataset_array

    return data

def hdf2dict(file_name):
    # Define the dictionary to store the datasets
    data = {}
    #file_name = 'macro421_UO2_03__900K.h5'
    with h5py.File("..//02.Macro.XS.421g/" + file_name, "r") as file:
        # Iterate over the dataset names in the group
        for dataset_name in file.keys():
            # Read the dataset
            dataset = file[dataset_name]
            # Check if the dataset is a struct
            if isinstance(dataset, h5py.Group):
                # Create a dictionary to store the struct fields
                struct_data = {}
                # Iterate over the fields in the struct
                for field_name in dataset.keys():
                    # Read the field dataset
                    field_dataset = np.array(dataset[field_name])
                    # Store the field dataset in the struct dictionary
                    struct_data[field_name] = field_dataset
                # Store the struct data in the main data dictionary
                data[dataset_name] = struct_data
            else:
                # Read the dataset as a regular array
                dataset_array = np.array(dataset)
                # Store the dataset array in the dictionary
                data[dataset_name] = dataset_array
    return data

def convert(solution):
    
    global g
    
    ng = 421
    # Define the cell array for angular fluxes
    fi = np.zeros((ng, g['N'], g['nNodesX'], g['nNodesY']))
    #solution = np.ones(261020)  # Assuming 261020 elements in the solution vector
    flux = solution.reshape(ng, -1)
    nEq = 0
    for iy in range(g['nNodesY']):
        for ix in range(g['nNodesX']):
            for n in range(g['N']):
                if g['muZ'][n] >= 0 and not (ix == 0 and g['muX'][n] > 0) and not \
                (ix == g['nNodesX'] - 1 and g['muX'][n] < 0) and not \
                (iy == 0 and g['muY'][n] > 0) and not \
                (iy == g['nNodesY'] - 1 and g['muY'][n] < 0):
                    fi[:, n, ix, iy] = flux[:, nEq]
                    nEq += 1

    for n in range(g['N']):
        if g['muZ'][n] < 0:
            fi[:, n] = fi[:, g['nRefZ'][n]]

    # Boundary conditions
    for n in range(g['N']):
        if g['muX'][n] > 0:
            fi[:, n, 0, :] = fi[:, g['nRefX'][n], 0, :]
        if g['muX'][n] < 0:
            fi[:, n, g['nNodesX'] - 1, :] = fi[:, g['nRefX'][n], g['nNodesX'] - 1, :]

    for n in range(g['N']):
        if g['muY'][n] > 0:
            fi[:, n, :, 0] = fi[:, g['nRefY'][n], :, 0]
        if g['muY'][n] < 0:
            fi[:, n, :, g['nNodesY'] - 1] = fi[:, g['nRefY'][n], :, g['nNodesY'] - 1]
   
    return fi

@nb.njit
def matmul(vector1, vector2):
    if vector1.ndim != 1 or vector2.ndim != 1:
        raise ValueError("Both inputs must be 1-dimensional arrays.")

    # Get the lengths of the vectors
    m = len(vector1)
    n = len(vector2)

    # Create a zero-filled 2D array of size m x n
    result = np.zeros((m, n))

    # Perform matrix multiplication (dot product) element-wise
    for i in range(m):
        for j in range(n):
            result[i, j] = vector1[i] * vector2[j]

    return result

def funDO(solution):
    # Convert 1D solution vector x to the cell array of angular flux fi
    fi = convert(solution)
    global g, SigT
    
    nEq = 0
    LHS_list = []

    for iy in range(g["nNodesY"]):
        for ix in range(g["nNodesX"]):
            for n in range(g["N"]):
                if g["muZ"][n] >= 0 and not ((ix == 0 and g["muX"][n] > 0) or (ix == g["nNodesX"] - 1 and g["muX"][n] < 0) or
                                             (iy == 0 and g["muY"][n] > 0) or (iy == g["nNodesY"] - 1 and g["muY"][n] < 0)):
                    # Gradients
                    dfidx, dfidy = gradients(n, ix, iy, fi)

                    nEq += 1
                    LHS_list.append(g["muX"][n] * dfidx + g["muY"][n] * dfidy + SigT[ix, iy] * fi[:, n, ix, iy])

    # Make 1D vector
    LHS = np.array(LHS_list)
    ax = LHS.reshape(-1)

    return ax

def gradients(n, ix, iy, fi):

    global g

    if g["muX"][n] > 0:
        if ix == 0:
            dfiX = fi[:, g["nRefX"][n] - 1, ix, iy] - fi[:, g["nRefX"][n] - 1, ix + 1, iy]
        else:  # if ix > 0
            dfiX = fi[:, n, ix, iy] - fi[:, n, ix - 1, iy]
    else:  # if g["muX"](n) <= 0
        if ix == g["nNodesX"] - 1:
            dfiX = fi[:, g["nRefX"][n] - 1, ix - 1, iy] - fi[:, g["nRefX"][n] - 1, ix, iy]
        else:  # if ix < g["nNodesX"] - 1
            dfiX = fi[:, n, ix + 1, iy] - fi[:, n, ix, iy]

    if g["muY"][n] > 0:
        if iy == 0:
            dfiY = fi[:, g["nRefY"][n] - 1, ix, iy] - fi[:, g["nRefY"][n] - 1, ix, iy + 1]
        else:  # if iy > 0
            dfiY = fi[:, n, ix, iy] - fi[:, n, ix, iy - 1]
    else:  # if g["muY"](n) <= 0
        if iy == g["nNodesY"] - 1:
            dfiY = fi[:, g["nRefY"][n] - 1, ix, iy - 1] - fi[:, g["nRefY"][n] - 1, ix, iy]
        else:  # if iy < g["nNodesY"] - 1
            dfiY = fi[:, n, ix, iy + 1] - fi[:, n, ix, iy]

    dfidx = dfiX / g["delta"]
    dfidy = dfiY / g["delta"]

    return dfidx, dfidy


# Start stopwatch

start_time = t.time()

# Global variables with geometry parameters, total cross section, and number of groups
#global g, SigT, ng
# input and initialize the geometry of the PWR unit cell (the function is in '..\00.Lib')
lib_path = os.path.join('..', '00.Lib')
file_path_PWR = os.path.join(lib_path, 'initPWR_like.h5')

if not os.path.exists(file_path_PWR):
    # File doesn't exist, call initPWR_like() function
    initPWR_like()
# Read in the necessary data struct. Options: {fr, g, th}
g = readPWR_like("g")
#--------------------------------------------------------------------------
# Path to macrosconp.pic cross section data:
#import os
#path_to_data = os.path.join('..', '02.Macro.XS.421g')
# Fill the structures fuel, clad, and cool with the cross-section data
fuel = hdf2dict('macro421_UO2_03__900K.h5')  # INPUT
print(f"File 'macro421_UO2_03__900K.h5' has been read in.")
clad = hdf2dict('macro421_Zry__600K.h5')     # INPUT
print(f"File 'macro421_Zry__600K.h5' has been read in.")
cool  = hdf2dict('macro421_H2OB__600K.h5')   # INPUT
print(f"File 'macro421_H2OB__600K.h5' has been read in.")

# Number of energy groups
ng = fuel["ng"]

#--------------------------------------------------------------------------
# Number of nodes
#g = {}
g['nNodesX'] = 10  # INPUT
g['nNodesY'] = 2   # INPUT

# Define the mesh step, nodes coordinates, and node volumes
g['delta'] = 0.2  # cm
volume = np.ones((g['nNodesX'], g['nNodesY'])) * g['delta']**2
volume[[0, -1], :] /= 2
volume[:, [0, -1]] /= 2

# Define the material for each node (0 is coolant, 1 is cladding, 2 is fuel)
mat = np.array([[2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 1, 0, 0, 0, 0]])

#--------------------------------------------------------------------------
# Path to Lebedev quadrature function:
# path_to_lebedev = os.path.join('..', '00.Lebedev')

# Number of discrete ordinates, an even integer (possible values are
# determined by the Lebedev quadratures: 6, 14, 26, 38, 50, 74, 86, 110,
# 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730,
# 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810):
g['N'] = 110  # INPUT


# Get leb["x"], leb["y"] and leb["z"] values for the g["N"] base points on a unit sphere
# as well as the associated weights leb["w"] (the unit sphere area corresponding
# to the base points, sum up to 4*np.pi) using the Lebedev quadrature rules.
#from getLebedevSphere import getLebedevSphere  # Assuming the function is available
leb = getLebedevSphere(g['N'])
g['muX'] = leb['x']
g['muY'] = leb['y']
g['muZ'] = leb['z']
g['W'] = leb['w']

# Find the reflective directions for X, Y, and Z directions
g['nRefX'] = np.zeros(g['N'], dtype=int)
g['nRefY'] = np.zeros(g['N'], dtype=int)
g['nRefZ'] = np.zeros(g['N'], dtype=int)

# Find the reflective directions for X, Y, and Z directions
for n in range(g['N']):
    for nn in range(g['N']):
        if np.allclose(g['muX'][nn], -g['muX'][n]) and np.allclose(g['muY'][nn], g['muY'][n]) and np.allclose(g['muZ'][nn], g['muZ'][n]):
            g['nRefX'][n] = nn
        if np.allclose(g['muY'][nn], -g['muY'][n]) and np.allclose(g['muX'][nn], g['muX'][n]) and np.allclose(g['muZ'][nn], g['muZ'][n]):
            g['nRefY'][n] = nn
        if np.allclose(g['muZ'][nn], -g['muZ'][n]) and np.allclose(g['muX'][nn], g['muX'][n]) and np.allclose(g['muY'][nn], g['muY'][n]):
            g['nRefZ'][n] = nn

#--------------------------------------------------------------------------
# Scattering source anisotropy: 0 -- P0 (isotropic), 1 -- P1
g['L'] = 0  # INPUT
# Initialize the 'R' key in the 'g' dictionary
g['R'] = {}
# Calculate spherical harmonics for every ordinate
for n in range(g['N']):
    g['R'][n] = np.zeros((2 * g['L'] + 1, 2 * g['L'] + 1))

    for jLgn in range(g['L'] + 1):
        for m in range(-jLgn, jLgn + 1):
            if jLgn == 0 and m == 0:
                g['R'][n][jLgn, jLgn + m] = 1
            elif jLgn == 1 and m == -1:
                g['R'][n][jLgn, jLgn + m] = g['muZ'][n]
            elif jLgn == 1 and m == 0:
                g['R'][n][jLgn, jLgn + m] = g['muX'][n]
            elif jLgn == 1 and m == 1:
                g['R'][n][jLgn, jLgn + m] = g['muY'][n]

#--------------------------------------------------------------------------
# Construct the cross sections
SigA, SigP, chi, SigT = np.zeros((10, 2, 421)), np.zeros((10, 2, 421)), np.zeros((10, 2, 421)), np.zeros((10, 2, 421))
Sig2 = np.zeros((10, 2, 421, 421))
SigS = np.zeros((jLgn+1, 10, 2, 421, 421))
for iy in range(g['nNodesY']):
    for ix in range(g['nNodesX']):
        if mat[iy, ix] == 2:  # fuel
            SigA[ix, iy] = fuel['SigF'] + fuel['SigC'] + fuel['SigL'] + np.sum(fuel["sig2_G"]["sparse_Sig2"], axis=1)
            for jLgn in range(g['L'] + 1):
                SigS[jLgn][ix, iy] = fuel['sigS_G'][f"sparse_SigS[{jLgn}]"]
            Sig2[ix, iy] = fuel["sig2_G"]["sparse_Sig2"]
            SigP[ix, iy] = fuel['SigP']
            chi[ix, iy] = fuel['chi']
        elif mat[iy, ix] == 1:  # cladding
            SigA[ix, iy] = clad['SigF'] + clad['SigC'] + clad['SigL'] + np.sum(clad["sig2_G"]["sparse_Sig2"], axis=1)
            for jLgn in range(g['L'] + 1):
                SigS[jLgn][ix, iy] = clad['sigS_G'][f"sparse_SigS[{jLgn}]"]
            Sig2[ix, iy] = clad["sig2_G"]["sparse_Sig2"]
            SigP[ix, iy] = clad['SigP']
            chi[ix, iy] = clad['chi']
        elif mat[iy, ix] == 0:  # coolant
            SigA[ix, iy] = cool['SigF'] + cool['SigC'] + cool['SigL'] + np.sum(cool["sig2_G"]["sparse_Sig2"], axis=1)
            for jLgn in range(g['L'] + 1):
                SigS[jLgn][ix, iy] = cool['sigS_G'][f"sparse_SigS[{jLgn}]"]
            Sig2[ix, iy] = cool["sig2_G"]["sparse_Sig2"]
            SigP[ix, iy] = cool['SigP']
            chi[ix, iy] = cool['chi']

        # Total cross section
        SigT[ix, iy] = SigA[ix, iy] + np.sum(SigS[0][ix, iy], axis=1)

#--------------------------------------------------------------------------
# Count the number of equations
nEq = 0
for iy in range(1, g['nNodesY'] + 1):
    for ix in range(1, g['nNodesX'] + 1):
        for n in range(1, g['N'] + 1):
            if g['muZ'][n - 1] >= 0 and not (ix == 1 and g['muX'][n - 1] > 0) and not   \
                (ix == g['nNodesX'] and g['muX'][n - 1] < 0) and not (iy == 1 and       \
                g['muY'][n - 1] > 0) and not (iy == g['nNodesY'] and g['muY'][n - 1] < 0):
                nEq = nEq + ng

#--------------------------------------------------------------------------
keff = []
residual = []

# Number of outer iterations
numIter = 200  # INPUT

# Set the initial flux equal 1.
solution = np.ones(nEq)

# Main iteration loop
for nIter in range(1, numIter + 1):
    #-----------------------------------------------------------------------
    # Make a guess for the solution
    # Just take the solution from the previous iteration as a guess
    guess = solution.copy()
    #-----------------------------------------------------------------------
    # Convert 1D guess vector to the array of angular flux fi
    fi = convert(guess)
    #-----------------------------------------------------------------------
    fiL = np.zeros((g['nNodesX'], g['nNodesY'], ng, g['L'] + 1, 2 * g['L'] + 1))
    FI = np.zeros((g['nNodesX'], g['nNodesY'], ng))

    for iy in range(1, g['nNodesY'] + 1):
        for ix in range(1, g['nNodesX'] + 1):
            for jLgn in range(g['L'] + 1):
                for m in range(-jLgn, jLgn + 1):
                    SUM = np.zeros(ng)
                    for n in range(g['N']):
                        SUM += fi[:, n, ix - 1, iy - 1] * g['R'][n][jLgn, jLgn + m] * g['W'][n]
                    fiL[ix - 1, iy - 1, :, jLgn, jLgn + m] = SUM
            FI[ix - 1, iy - 1, :] = fiL[ix - 1, iy - 1, :, 0, g['L']]

    #-----------------------------------------------------------------------
    # pRate is total neutron production rate
    pRate = 0
    # aRate is total neutron absorption rate
    aRate = 0
    ans1 = np.zeros((10,421))
    ans2 = np.zeros((10,421))
    ans3 = np.zeros((10, 2))
    for iy in range(g['nNodesY']):
        for ix in range(g['nNodesX']):
            pRate += (SigP[ix, iy] + 2 * np.sum(Sig2[ix, iy, :], axis = 1)) @ FI[ix, iy] * volume[ix, iy]
            #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ The exact point where everything goes to shit
            # Best theory at the moment is that floating point errors start to accumulate a will start causing noticable 
            # differences in final output
            aRate += SigA[ix, iy] @ FI[ix, iy] * volume[ix, iy]
            #print("pRate", pRate, "\naRate =", aRate)
    #print(sp.sparse.coo_matrix(2 * np.sum(Sig2[ix, iy, :], axis = 1)))

    #print("aRate", aRate, "\npRate =", pRate)

    # We evaluate the multiplication factor as a ratio of neutron
    # production rate and neutron absorption rate (there is no neutron
    # leakage in the infinite lattice):
    keff.append(pRate / aRate)
    print(f'keff = {keff[-1]:9.5f} #nOuter = {nIter:3}', end=' ')

    #-----------------------------------------------------------------------
    # Calculate fission, (n,2n) and scattering neutron sources
    # Initialize the total neutron source vector and nEq
    nEq = 0
    q2 = np.zeros(ng)
    #qT_list = np.zeros((ng, ))
    qT_list = []

    for iy in range(g["nNodesY"]):
        for ix in range(g["nNodesX"]):
            # Fission source (1/s-cm3-steradian)
            qF = matmul(chi[ix, iy], SigP[ix, iy]) @ FI[ix, iy] / keff[-1] / (4*np.pi)

            # Isotropic source from (n,2n) (1/s-cm3-steradian)
            q2 = 2 * np.dot(np.transpose(Sig2[ix, iy]), FI[ix, iy]) / (4*np.pi)

            for n in range(g["N"]):
                if g["muZ"][n] >= 0 and not ((ix == 0 and g["muX"][n] > 0) or (ix == g["nNodesX"]-1 and g["muX"][n] < 0) or
                                            (iy == 0 and g["muY"][n] > 0) or (iy == g["nNodesY"]-1 and g["muY"][n] < 0)):
                    # Scattering source (1/s-cm3-steradian), isotropic (g["L"] = 0) or anisotropic (g["L"] > 0)
                    qS = np.zeros(ng)
                    for jLgn in range(g["L"] + 1):
                        SUM = np.zeros(ng)
                        for m in range(-jLgn, jLgn + 1):
                            SUM += fiL[ix, iy][:, jLgn, jLgn + m] * g["R"][n][jLgn, jLgn + m]
                        qS += (2*jLgn+1) * np.dot(np.transpose(SigS[jLgn][ix, iy]), SUM) / (4*np.pi)

                    nEq += 1
                    # Right-hand side is a total neutron source:
                    qT_list.append(qF + q2 + qS)
                    #np.append(qT_list, qF + q2 + qS)
                    #qT_list[:,nEq] = qF + q2 + qS

    # Convert the list of qT to a numpy array
    qT = np.array(qT_list)

    # Reshape qT into a column vector
    #RHS = qT.reshape(-1, 1)
    #RHS.shape
    dim = qT.shape[0] * qT.shape[1]
    RHS = np.zeros(dim)
    for i in range(qT.shape[0]):
        for j in range(qT.shape[1]):
            RHS[i*qT.shape[1] + j] = qT[i][j] 

    #-----------------------------------------------------------------------
    # Relative residual reduction factor
    errtol = 1.e-4;                                                      # INPUT
    # maximum number of iterations
    maxit = 2000;                                                        # INPUT
    # Solver of a system of linear algebraic equations:
    A = sp.sparse.linalg.LinearOperator((len(RHS), len(RHS)), matvec=lambda x: funDO(x))
    # Call the bicgstab solver
    solution, info = sp.sparse.linalg.bicgstab(A, RHS, tol=1e-4, maxiter=2000, x0=guess)

    # Compute nInner based on the number of iterations performed by the solver
    #nInner = info['iterations']

    # Compute the residual and add it to the list
    matAx = A @ solution
    nInner = np.linalg.norm(RHS - matAx)/np.linalg.norm(RHS)
    residual.append(nInner)

    print(
        #f'nInner = {nInner + 1}, 
        f'nInner = {nInner:.5e}, target = {errtol:.5e}')

    if nInner <= errtol:
        break

stop_time = t.time()
print(f'Elapsed time: {stop_time-start_time}')
print("Solution:", solution)