
import numpy as np
from optparse import OptionParser


import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

from time import time


__version__ = "0.0.1"

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

def readDOSFromFile(opts):
    '''
    Read DOS info from file.

    the format of the DOS file:
    first line: labels of the dos: # eng dos occup e_sum
    '''
    
    inp = open(opts.dos).readlines()
    labels = inp[0].split()[1:]
    
    # Data
    DOS = np.array([line.split() for line in inp[1:] if line.strip()],
                    dtype=float)
    
    xen = DOS[:, 0]
    tdos = DOS[:, 1]
    occ = DOS[:, 2]
    e_sum = DOS[:,3]
    
    opts.pdosLabel = labels

    return xen, tdos, occ, e_sum

def dosplot(xen, tdos, opts):
    '''
    Use matplotlib to plot density of state
    '''

    width, height = opts.figsize
    xmin, xmax = opts.xlim

    plt.style.use(opts.mpl_style)
    
    # DO NOT use unicode minus regardless of the style
    mpl.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    fig.set_size_inches(width, height)
    ax = plt.subplot(111)

    LINES = []

    plabels = []

    plabels += ['total']

    xen -= opts.zero # energy reference of the band plot

    if opts.showtotal:
            
        line, im = gradient_fill(xen, tdos, ax=ax,
                                 color = 'black',
                                 lw=0.5
                                 )
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

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.legend(LINES, plabels,
              loc=opts.legendloc,
              fontsize='small',
              frameon=True,
              framealpha=0.6)

    plt.tight_layout(pad=0.50)
    plt.savefig(opts.dosimage, dpi=opts.dpi)
    


# Read Commands Line
opts, args = command_line_arg()

xen, tdos, occ, e_sum = readDOSFromFile(opts)

t0 = time()
dosplot(xen, tdos, opts)
t1 = time()

print('DOS plot completed! Time Used: %.2f [sec]' % (t1 - t0))



