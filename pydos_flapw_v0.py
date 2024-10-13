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

    par.add_option("-i", '--input',
                   action='store', type="string", dest='dos',
                   default='DOS',
                   help='location of the DOS file')
    
    return par.parse_args()

def dosplot(xen, tdos, pdos, opts):
    '''
    Use matplotlib to plot band structure
    '''

    width, height = opts.figsize
    xmin, xmax = opts.xlim
    dpi = opts.dpi

    plt.style.use(opts.mpl_style)
    # DO NOT use unicode minus regardless of the style
    mpl.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    fig.set_size_inches(width, height)
    ax = plt.subplot(111)

    LINES = []
    nspin = tdos.shape[1]

    plabels = []
    LWs = []
    LCs = []
    if opts.pdosAtom:
        plabels = ['p_%d' % ii for ii in range(len(opts.pdosAtom))]
        LWs = [0.5 for ii in range(len(opts.pdosAtom))]
        LCs = [None for ii in range(len(opts.pdosAtom))]

        for ii in range(min(len(opts.pdosAtom), len(opts.pdosLabel))):
            plabels[ii] = opts.pdosLabel[ii]
        for ii in range(min(len(opts.pdosAtom), len(opts.linewidth))):
            LWs[ii] = opts.linewidth[ii]
        for ii in range(min(len(opts.pdosAtom), len(opts.linecolors))):
            LCs[ii] = opts.linecolors[ii]

    plabels += ['total']

    xen -= opts.zero
    for ip, p in enumerate(pdos):
        for ii in range(nspin):
            fill_direction = 1 if ii == 0 else -1
            lc = LCs[ip] if ii == 0 else line.get_color()

            if opts.fill:
                line, im = gradient_fill(xen, p[:, ii], ax=ax, lw=LWs[ip],
                                         color=lc,
                                         direction=fill_direction)
            else:
                line, = ax.plot(xen, p[:, ii], lw=LWs[ip], alpha=0.6,
                                color=lc)
            if ii == 0:
                LINES += [line]

    if opts.showtotal:
        for ii in range(nspin):
            fill_direction = 1 if ii == 0 else -1
            lc = 'k' if ii == 0 else line.get_color()

            if opts.fill:
                line, im = gradient_fill(xen, tdos[:, ii], ax=ax,
                                         color=lc,
                                         lw=0.5,
                                         # zorder=-1,
                                         direction=fill_direction,
                                         )
            else:
                line, = ax.plot(xen, tdos[:, ii], color=lc,
                                lw=0.5, alpha=0.6)
            if ii == 0:
                LINES += [line]

    ax.set_xlabel('Energy [eV]',  # fontsize='small',
                  labelpad=5)
    ax.set_ylabel('DOS [arb. unit]',  # fontsize='small',
                  labelpad=10)
    ax.tick_params(which='both', labelsize='small')

    ax.set_xlim(xmin, xmax)
    if opts.ylim is not None:
        ymin, ymax = opts.ylim
        ax.set_ylim(ymin, ymax)

    # ax.set_yticklabels([])

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    opts.pdosLabel = plabels
    ax.legend(LINES, plabels,
              loc=opts.legendloc,
              fontsize='small',
              frameon=True,
              framealpha=0.6)

    plt.tight_layout(pad=0.50)
    plt.savefig(opts.dosimage, dpi=opts.dpi)
    
def readDOSFromFile(opts):
    '''
    Read DOS info from file.

    the format of the DOS file:
    first line: ISPIN, NEDOS
    second line: labels of the dos
    next lines:
        if ISPIN = 1:
            Energy pDOS1 PDOS2 ... TotalDOS
        else:
            Energy pDOS1_up PDOS2_up ... TotalDOS_up pDOS1_down PDOS2_down ... TotalDOS_down
    '''

    inp = open(opts.dos).readlines()
    # inp = open("dos_nonpolarized").readlines()

    # the dos basic info
    # nspin: 1 -> nonpolarized
    # nspin: 2 -> polarized
    nspin, nedos = [int(x) for x in inp[0].split()[1:]]
    # nspin, nedos = [int(x) for x in inp[0].split()[:2]]
    labels = inp[1].split()[1:]
    # data
    DOS = np.array([line.split() for line in inp[2:] if line.strip()],
                   dtype=float)
    NoPdos = (DOS.shape[1] - 1) // nspin - 1

    tDOS = np.empty((nedos, nspin))
    pDOS = []
    xen = DOS[:, 0]
    for ii in range(nspin):
        tDOS[:, ii] = DOS[:, (ii + 1) * (NoPdos + 1)]
    for pp in range(NoPdos):
        tmp = []
        for ii in range(nspin):
            tmp += [DOS[:, (pp + 1) + ii * (NoPdos + 1)]]
        pDOS += [np.array(tmp).T]

    opts.nedos = nedos
    opts.pdosAtom = ['' for x in range(NoPdos)]
    opts.pdosLabel = labels

    return xen, tDOS, pDOS


if __name__ == '__main__':
    from time import time

    __version__ = "1.0"

    # Read Commands Line
    opts, args = command_line_arg()

    xen, tdos, pdos = readDOSFromFile(opts)

    print(xen, tdos, pdos)

    # t0 = time()
    # dosplot(xen, tdos, pdos, opts)
    # t1 = time()
    # print('DOS plot completed! Time Used: %.2f [sec]' % (t1 - t0))

    

