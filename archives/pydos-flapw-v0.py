#!/usr/bin/env python

from __future__ import print_function

import os, re
import numpy as np
from ase.io import read
from optparse import OptionParser


import matplotlib as mpl

mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False

'''
mpl.use('agg'): Configures matplotlib to use a file-based backend, rendering plots directly into image files rather than displaying them interactively.
mpl.rcParams['axes.unicode_minus'] = False: Ensures that the ASCII minus sign (-) is used in plots, avoiding potential issues with Unicode characters.
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option("-i", '--input',
                   action='store', type="string", dest='dos',
                   default='dos.xy',
                   help='location of the DOS file')
    
    par.add_option('-s', '--size', nargs=2,
                   action='store', type="float", dest='figsize',
                   default=(4.8, 3.0),
                   help='figure size of the output plot')
    
    par.add_option('-x', nargs=2,
                   action='store', type="float", dest='xlim',
                   default=(-6, 6),
                   help='x limit of the dos plot')
    
    par.add_option('--dpi',
                   action='store', type="int", dest='dpi',
                   default=360,
                   help='resolution of the output image')
    
    par.add_option('--style',
                   action='store', type='string', dest='mpl_style',
                   default='default',
                   help='plot style of matplotlib. See "plt.style.available" for list of available styles.')
    
    par.add_option('-z', '--zero',
                   action='store', type="float",
                   dest='zero', default=0.0,
                   help='energy reference of the band plot')
    
    par.add_option('-y', nargs=2,
                   action='store', type="float", dest='ylim',
                   default=None,
                   help='energy range of the band plot')
    
    par.add_option('--lloc',
                   action='store', type="string", dest='legendloc',
                   default='upper right',
                   help='legend location of dos plot')
    
    par.add_option('-o', '--output',
                   action='store', type="string", dest='dosimage',
                   default='dos.png',
                   help='output image name, "dos.png" by default')
    
    par.add_option('--fill',
                   action='store_true', dest='fill',
                   default=True,
                   help='fill under the DOS')
    
    par.add_option('--tot',
                   action='store_true', dest='showtotal',
                   default=True,
                   help='show total dos')

    par.add_option('--notot',
                   action='store_false', dest='showtotal',
                   help='not show total dos')
    
    
    return par.parse_args()


def gradient_fill(x, y, fill_color=None, ax=None, direction=1, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    # print fill_color
    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    if direction == 1:
        z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    else:
        z[:, :, -1] = np.linspace(alpha, 0, 100)[:, None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    if direction == 1:
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    else:
        xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
    clip_path = Polygon(xy, lw=0.0, facecolor='none',
                        edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)

    return line, im
############################################################

    

# def dosplot(xen, tdos, pdos, opts):
def dosplot(xen, tdos, opts):
    '''
    Use matplotlib to plot density of state
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
    # nspin = tdos.shape[1]
    nspin = tdos.shape[0]

    plabels = []
    LWs = []
    LCs = []
    
    # Adding Projected Density of State, for each orbital
    
    # if opts.pdosAtom:
    #     plabels = ['p_%d' % ii for ii in range(len(opts.pdosAtom))]
    #     LWs = [0.5 for ii in range(len(opts.pdosAtom))]
    #     LCs = [None for ii in range(len(opts.pdosAtom))]

    #     for ii in range(min(len(opts.pdosAtom), len(opts.pdosLabel))):
    #         plabels[ii] = opts.pdosLabel[ii]
    #     for ii in range(min(len(opts.pdosAtom), len(opts.linewidth))):
    #         LWs[ii] = opts.linewidth[ii]
    #     for ii in range(min(len(opts.pdosAtom), len(opts.linecolors))):
    #         LCs[ii] = opts.linecolors[ii]

    plabels += ['total']

    xen -= opts.zero
    # for ip, p in enumerate(pdos):
    #     for ii in range(nspin):
    #         fill_direction = 1 if ii == 0 else -1
    #         lc = LCs[ip] if ii == 0 else line.get_color()

    #         if opts.fill:
    #             line, im = gradient_fill(xen, p[:, ii], ax=ax, lw=LWs[ip],
    #                                      color=lc,
    #                                      direction=fill_direction)
    #         else:
    #             line, = ax.plot(xen, p[:, ii], lw=LWs[ip], alpha=0.6,
    #                             color=lc)
    #         if ii == 0:
    #             LINES += [line]

    if opts.showtotal:
        line, im = gradient_fill(xen, tdos, ax=ax,
                                 color = 'black',
                                 lw=0.5
                                 # zorder=-1
                                 )
        LINES += [line]
        # for ii in range(nspin):
        #     fill_direction = 1 if ii == 0 else -1
            # lc = 'k' if ii == 0 else line.get_color()

            # if opts.fill:
            #     # line, im = gradient_fill(xen, tdos[:, ii], ax=ax,
            #     #                          color=lc,
            #     #                          lw=0.5,
            #     #                          # zorder=-1,
            #     #                          direction=fill_direction,
            #     #                          )
            #     ine, im = gradient_fill(xen, tdos[ii], ax=ax,
            #                              lw=0.5,
            #                              # zorder=-1,
            #                              direction=fill_direction,
            #                              )
            # else:
            #     # line, = ax.plot(xen, tdos[:, ii], color=lc,
            #     #                 lw=0.5, alpha=0.6)
            #     line, = ax.plot(xen, tdos[:, ii],
            #                     lw=0.5, alpha=0.6)
            # if ii == 0:
            #     LINES += [line]

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
    first line: labels of the dos: # eng dos occup e_sum
    '''
    
    # This needs to change later
    # inp = open("dos.xy").readlines() 
    inp = open(opts.dos).readlines()
    labels = inp[0].split()[1:]
    
    # Data
    DOS = np.array([line.split() for line in inp[1:] if line.strip()],
                    dtype=float)

    # NoPdos = (DOS.shape[1] - 1) // nspin - 1

    # tDOS = np.empty((nedos, nspin))
    # pDOS = []
    
    xen = DOS[:, 0]
    tDOS = DOS[:, 1]
    
    # for ii in range(nspin):
    #     tDOS[:, ii] = DOS[:, (ii + 1) * (NoPdos + 1)]
    # for pp in range(NoPdos):
    #     tmp = []
    #     for ii in range(nspin):
    #         tmp += [DOS[:, (pp + 1) + ii * (NoPdos + 1)]]
    #     pDOS += [np.array(tmp).T]

    # opts.nedos = nedos
    # opts.pdosAtom = ['' for x in range(NoPdos)]
    opts.pdosLabel = labels

    # return xen, tDOS, pDOS

    return xen, tDOS


if __name__ == '__main__':
    from time import time

    __version__ = "1.0"

    # Read Commands Line
    opts, args = command_line_arg()

    # xen, tdos, pdos = readDOSFromFile(opts)
    
    xen, tdos = readDOSFromFile(opts)

    t0 = time()
    # dosplot(xen, tdos, pdos, opts)
    dosplot(xen, tdos, opts)

    t1 = time()
    print('DOS plot completed! Time Used: %.2f [sec]' % (t1 - t0))

    

