#!/usr/bin/env python

from __future__ import print_function

import os, re
import numpy as np
from ase.io import read
from optparse import OptionParser


import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option('--tot',
                   action='store_true', dest='showtotal',
                   default=True,
                   help='show total dos')


if __name__ == '__main__':
    from time import time

    # Read Commands Line
    opts, args = command_line_arg()

