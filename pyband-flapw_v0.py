import os
import re
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

def kpath_name_parse(KPATH_STR):
    '''
    Parse the kpath string
    '''

    KPATH_STR = KPATH_STR.upper()
    # legal kpath separators: blank space, comma, hypen or semicolon
    KPATH_SEPARATORS = ' ,-;'
    # Greek Letters Dictionaries
    GREEK_KPTS = {
        'G':      r'$\mathrm{\mathsf{\Gamma}}$',
        'GAMMA':  r'$\mathrm{\mathsf{\Gamma}}$',
        'DELTA':  r'$\mathrm{\mathsf{\Delta}}$',
        'LAMBDA': r'$\mathrm{\mathsf{\Lambda}}$',
        'SIGMA':  r'$\mathrm{\mathsf{\Sigma}}$',
    }

    # If any of the kpath separators is in the kpath string
    if any([s in KPATH_STR for s in KPATH_SEPARATORS]):
        kname = [
            GREEK_KPTS[x] if x in GREEK_KPTS else 
	    r'$\mathrm{{\mathsf{{{}}}}}$'.format(x)
            for x in re.sub('['+KPATH_SEPARATORS+']', ' ', KPATH_STR).split()
        ]
    else:
        kname = [
            GREEK_KPTS[x] if x in GREEK_KPTS else 
	    r'$\mathrm{{\mathsf{{{}}}}}$'.format(x)
            for x in KPATH_STR
        ]

    return kname

def parseList(string):
    def parseRange(rng):
        # print(rng)
        m = re.match(r'(\d+)(?:[-:](\d+))?(?:[-:](\d+))?$', rng)
        if not m:
            raise ValueError(
                """
The index should be assigned with combination of the following ways:
-> 10, a single band with index 10
-> 20:30, or '20-30', a continuous range from 20 to 30, 30 included
-> 30:50:2, or '30-50:2', a continues range from 30 to 50, with step size 2
-> '1 2 3', all the patterns above in one string separated by spaces.
For example: '1 4:6 8:16:2' will be converted to '1 4 5 6 8 10 12 14 16'
"""
            )
        ii = m.group(1)
        jj = m.group(2) or ii
        ss = m.group(3) or 1
        return [x-1 for x in range(int(ii), int(jj)+1, int(ss))]

    ret = []
    for rng in string.split():
        ret += parseRange(rng)
    return list(set(ret))

def parseSpdProjection(spd):
    '''
    Parse spdProjections string.  str -> [int]

    # Ionizing
    '''
    spd_dict = {
            's'   : [0],
            'p'   : [1, 2, 3],
            'd'   : [4, 5, 6, 7, 8],
            'f'   : [9, 10, 11, 12, 13, 14, 15],
            'py'  : [1],
            'pz'  : [2],
            'px'  : [3],
            'dxy' : [4],
            'dyz' : [5],
            'dz2' : [6],
            'dxz' : [7],
            'dx2' : [8],
    "fy(3x2-y2)"  : [9],
    "fxyz  "      : [10],
    "fyz2  "      : [11],
    "fz3   "      : [12],
    "fxz2  "      : [13],
    "fz(x2-y2)"   : [14],
    "fx(x2-3y2) " : [15],
    }

    ret = []
    for l in spd.split():
        try:
            assert int(l) <= 15, "Maximum spd index should be <= 15."
            ret += [int(l)]
        except:
            if l.lower() not in spd_dict:
                raise ValueError(
                   "Spd-projected wavefunction character of each KS orbital.\n"
                   "    s orbital: 0\n"
                   "    py, pz, px orbital: 1 2 3\n"
                   "    dxy, dyz, dz2, dxz, dx2 orbital: 4 5 6 7 8 \n"
                   "    fy(3x2-y2), fxyz, fyz2, fz3, fxz2, fz(x2-y2), fx(x2-3y2) orbital: 9 10 11 12 13 14 15\n"
                   "\nFor example, --spd 's dxy 10' specifies the s/dxy/fxyz components\n"
                )
            ret += spd_dict[l]

    return list(set(ret))

def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option('-f', '--file',
                   action='store', type="string",
                   dest='filename', default='band.xy',
                   help='location of OUTCAR')

    par.add_option('--procar',
                   action='store', type="string", dest='procar',
                   default='PROCAR',
                   help='location of the PROCAR')

    par.add_option('-z', '--zero',
                   action='store', type="float",
                   dest='efermi', default=None,
                   help='energy reference of the band plot')

    par.add_option('-o', '--output',
                   action='store', type="string", dest='bandimage',
                   default='band.png',
                   help='output image name, "band.png" by default')

    par.add_option('-k', '--kpoints',
                   action='store', type="string", dest='kpts',
                   default=None,
                   help='kpoint path, use ",", " ", ";" or "-" to separate the K-point if necessary.')

    par.add_option('--hse',
                   action='store_true', dest='isHSE',
                   default=False,
                   help='whether the calculation is HSE')

    par.add_option('--skip_kpts',
                   action='store', type="string", dest='skip_kpts',
                   default=None,
                   help='Skip the bands of the redundant k-points, usefull in HSE band plot.')

    par.add_option('--nseg',
                   action='append', type="int", dest='nseg',
                   default=[],
                   help='Number of kpoints in each segment, used with --skip_kpts.')

    par.add_option('-s', '--size', nargs=2,
                   action='store', type="float", dest='figsize',
                   default=(3.0, 4.0),
                   help='figure size of the output plot')

    par.add_option('-y', nargs=2,
                   action='store', type="float", dest='ylim',
                   default=(-3, 3),
                   help='energy range of the band plot')

    par.add_option('--hline',
                   action='append', type="float", dest='hlines',
                   default=[],
                   help='Add horizontal lines to the figure.')

    par.add_option('--vline',
                   action='append', type="float", dest='vlines',
                   default=[],
                   help='Add vertical lines to the figure.')

    par.add_option('--save_gnuplot',
                   action='store_true', dest='gnuplot',
                   default=False,
                   help='save output band energies in gnuplot format')

    par.add_option('--lw',
                   action='store', type="float", dest='linewidth',
                   default=1.0,
                   help='linewidth of the band plot')

    par.add_option('--lc',
                   action='store', type="str", dest='linecolors',
                   default=None,
                   help='line colors of the band plot')

    par.add_option('--dpi',
                   action='store', type="int", dest='dpi',
                   default=360,
                   help='resolution of the output image')

    par.add_option('--occ',
                   action='append', type="string", dest='occ',
                   default=[],
                   help='orbital contribution of each KS state')

    par.add_option('--occL',
                   action='store_true', dest='occLC',
                   default=False,
                   help='use Linecollection or Scatter to show the orbital contribution')

    par.add_option('--occLC_cmap',
                   action='store', type='string', dest='occLC_cmap',
                   default='jet',
                   help='colormap of the line collection')

    par.add_option('--occLC_lw',
                   action='store', type='float', dest='occLC_lw',
                   default=2.0,
                   help='linewidth of the line collection')

    par.add_option('--occLC_cbar_pos',
                   action='store', type='string', dest='occLC_cbar_pos',
                   default='top',
                   help='position of the colorbar')

    par.add_option('--occLC_cbar_ticks',
                   action='store', type='string', dest='occLC_cbar_ticks',
                   default=None,
                   help='ticks for the colorbar')

    par.add_option('--occLC_cbar_vmin',
                   action='store', type='float', dest='occLC_cbar_vmin',
                   default=None,
                   help='minimum value for the color plot')

    par.add_option('--occLC_cbar_vmax',
                   action='store', type='float', dest='occLC_cbar_vmax',
                   default=None,
                   help='maximum value for the color plot')

    par.add_option('--occLC_cbar_ticklabels',
                   action='store', type='string', dest='occLC_cbar_ticklabels',
                   default=None,
                   help='tick labels for the colorbar')

    par.add_option('--occLC_cbar_size',
                   action='store', type='string', dest='occLC_cbar_size',
                   default='3%',
                   help='size of the colorbar, relative to the axis')

    par.add_option('--occLC_cbar_pad',
                   action='store', type='float', dest='occLC_cbar_pad',
                   default=0.02,
                   help='pad between colorbar and axis')

    par.add_option('--occM',
                   action='append', type="string", dest='occMarker',
                   default=[],
                   help='the marker used in the plot')

    par.add_option('--occMs',
                   action='append', type="int", dest='occMarkerSize',
                   default=[],
                   help='the size of the marker')

    par.add_option('--occMc',
                   action='append', type="string", dest='occMarkerColor',
                   default=[],
                   help='the color of the marker')

    par.add_option('--spd',
                   action='append', type="string", dest='spdProjections',
                   default=[],
                   help='Spd-projected wavefunction character of each KS orbital.')

    par.add_option('--spin', action='store', dest='spin',
                   default=None, choices=['x', 'y', 'z'],
                   help='show the magnetization mx/y/z constributions to the states. Use this option along with --occ.')

    par.add_option('--lsorbit',
                   action='store_true', dest='lsorbit',
                   help='Spin orbit coupling on, special treament of PROCAR')

    par.add_option('-q', '--quiet',
                   action='store_true', dest='quiet',
                   help='not show the resulting image')

    return par.parse_args()

def bandplot(kpath, bands, efermi, kpt_bounds, opts, whts=None):
    '''
    Use matplotlib to plot band structure
    '''

    width, height = opts.figsize
    ymin, ymax = opts.ylim
    dpi = opts.dpi

    fig = plt.figure()
    fig.set_size_inches(width, height)
    ax = plt.subplot(111)

    # nspin, nkpts, nbands = bands.shape

    clrs = ['r', 'b']

    if opts.occLC and (whts is not None):
        from matplotlib.collections import LineCollection
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        LW = opts.occLC_lw
        DELTA = 0.3
        EnergyWeight = whts[0]
        norm = mpl.colors.Normalize(
                vmin = opts.occLC_cbar_vmin if opts.occLC_cbar_vmin else EnergyWeight.min(),
                vmax = opts.occLC_cbar_vmax if opts.occLC_cbar_vmax else EnergyWeight.max(),
        )
        # norm = mpl.colors.Normalize(0, 1)
        # create a ScalarMappable and initialize a data structure
        s_m = mpl.cm.ScalarMappable(cmap=opts.occLC_cmap, norm=norm)
        s_m.set_array([EnergyWeight])

        for Ispin in range(nspin):

            # If x and/or y are 2D arrays a separate data set will be drawn for
            # every column. If both x and y are 2D, they must have the same
            # shape. If only one of them is 2D with shape (N, m) the other must
            # have length N and will be used for every data set m.

            ax.plot(kpath, bands[Ispin],
                    lw=LW + 2 * DELTA,
                    color='gray', zorder=1)

            for jj in range(nbands):
                x = kpath
                y = bands[Ispin, :, jj]
                z = EnergyWeight[Ispin, :, jj]

                # ax.plot(x, y,
                #         lw=LW + 2 * DELTA,
                #         color='gray', zorder=1)

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments,
                                    # cmap=opts.occLC_cmap, # alpha=0.7,
                                    colors=[s_m.to_rgba(ww)
                                            for ww in (z[1:] + z[:-1])/2.]
                                    # norm=plt.Normalize(0, 1)
                                    )
                # lc.set_array((z[1:] + z[:-1]) / 2)
                lc.set_linewidth(LW)
                ax.add_collection(lc)

        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes(opts.occLC_cbar_pos.lower(),
                                      size=opts.occLC_cbar_size, pad=opts.occLC_cbar_pad)

        if opts.occLC_cbar_pos.lower() == 'top' or opts.occLC_cbar_pos.lower() == 'bottom':
            ori = 'horizontal'
        else:
            ori = 'vertical'
        cbar = plt.colorbar(s_m, cax=ax_cbar,
                            # ticks=[0.0, 1.0],
                            orientation=ori)
        if opts.occLC_cbar_ticks:
            cbar.set_ticks([
                float(x) for x in
                opts.occLC_cbar_ticks.split()
            ])
            if opts.occLC_cbar_ticklabels:
                cbar.set_ticklabels(opts.occLC_cbar_ticklabels.split())

        if ori == 'horizontal':
            cbar.ax.xaxis.set_ticks_position('top')
        else:
            cbar.ax.yaxis.set_ticks_position('right')


    else:
        # for Ispin in range(nspin):
        #     # If x and/or y are 2D arrays a separate data set will be drawn for
        #     # every column. If both x and y are 2D, they must have the same
        #     # shape. If only one of them is 2D with shape (N, m) the other must
        #     # have length N and will be used for every data set m.
        #     ax.plot(kpath, bands[Ispin],
        #             lw=opts.linewidth, color=opts.linecolors[Ispin],
        #             alpha=0.8, zorder=0)

        #     if whts is not None:
        #         kpath_x = np.tile(kpath, (nbands, 1)).T
        #         for ii in range(len(opts.occ)):
        #             ax.scatter(kpath_x, bands[Ispin],
        #                        color=opts.occMarkerColor[ii],
        #                        s=whts[ii][Ispin] *
        #                        opts.occMarkerSize[ii],
        #                        marker=opts.occMarker[ii], zorder=1, lw=0.0,
        #                        alpha=0.5)
        
        
        ax.plot(kpath, bands,
                lw=opts.linewidth, color='b',
                alpha=0.8, zorder=0)

        if whts is not None:
            kpath_x = np.tile(kpath, (nbands, 1)).T
            for ii in range(len(opts.occ)):
                ax.scatter(kpath_x, bands,
                           color=opts.occMarkerColor[ii],
                           s=whts[ii] *
                           opts.occMarkerSize[ii],
                           marker=opts.occMarker[ii], zorder=1, lw=0.0,
                           alpha=0.5)

            # for Iband in range(nbands):
            #     # if Iband == 0 else line.get_color()
            #     lc = opts.linecolors[Ispin]
            #     line, = ax.plot(kpath, bands[Ispin, :, Iband], lw=opts.linewidth, zorder=0,
            #                     alpha=0.8,
            #                     color=lc,
            #                     )
            #     if whts is not None:
            #         for ii in range(len(opts.occ)):
            #             ax.scatter(kpath, bands[Ispin, :, Iband],
            #                        color=opts.occMarkerColor[ii],
            #                        s=whts[ii][Ispin, :, Iband] *
            #                        opts.occMarkerSize[ii],
            #                        marker=opts.occMarker[ii], zorder=1, lw=0.0,
            #                        alpha=0.5)

    # for bd in kpt_bounds:
    #     ax.axvline(x=bd, ls='-', color='k', lw=0.5, alpha=0.5)

    # add extra horizontal/vertical lines
    for xx in opts.hlines:
        ax.axhline(y=xx, ls=':', color='k', lw=0.5, alpha=0.5)
    for yy in opts.vlines:
        ax.axvline(x=yy, ls=':', color='k', lw=0.5, alpha=0.5)

    ax.set_ylabel('Energy [eV]',  # fontsize='small',
                  labelpad=5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(kpath.min(), kpath.max())

    # ax.set_xticks(kpt_bounds)
    if opts.kpts:
        ax.set_xticklabels(kpath_name_parse(opts.kpts))
    else:
        ax.set_xticklabels([])

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tight_layout(pad=0.20)
    plt.savefig(opts.bandimage, dpi=opts.dpi)



def get_bandInfo(in_file='band.xy'):
    inp = open('band.xy').readlines()
    labels = inp[0].split()[1:]

    # Data
    BAND = np.array([line.split() for line in inp[1:] if line.strip()], dtype=float) 

    # kx = BAND[:, 0]
    # ky = BAND[:, 1]
    # kz = BAND[:, 2]
    # dist = BAND[:, 3]
    # eig = BAND[:, 4]
    # iband = BAND[:, 5]
    # line = BAND[:, 6]

    efermi = None
    wkpts = None
    kpt_bounds = None

    dist = BAND[:, [3,5]]
    kpath = dist
    
    # Find the unique values in the second column
    unique = np.unique(kpath[:,1])

    # Create a dictionary to store separated arrays based on second column values
    sep_ar = {val: kpath[kpath[:,1] == val] for val in unique}
    new_col = []

    # Remove the second column and collect the first column data
    for key, val in sep_ar.items():
        # Extract the first column and append to the list
        new_col.append(val[:,0])

    # Convert the list of arrays into a single new numpy array by column stacking
    kpath = np.column_stack(new_col)
    
    ######################################################

    bands = BAND[:, 4:5+1]

    # Find the unique values in the second column
    unique = np.unique(bands[:,1])

    # Create a dictionary to store separated arrays based on second column values
    sep_ar = {val: bands[bands[:,1] == val] for val in unique}
    new_col = []

    # Remove the second column and collect the first column data
    for key, val in sep_ar.items():
        # Extract the first column and append to the list
        new_col.append(val[:,0])

    # Convert the list of arrays into a single new numpy array by column stacking
    bands = np.column_stack(new_col)
    # bands = bands.reshape(1,2,-1)
    
    #######################################################
    
    # get band path
    # vkpt_diff = np.diff(vkpts, axis=0)
    # kpt_path = np.zeros(nkpts, dtype=float)
    # kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
    # # kpt_path /= kpt_path[-1]

    # # get boundaries of band path
    # xx = np.diff(kpt_path)
    # kpt_bounds = np.concatenate(
    #     ([0.0, ], kpt_path[1:][np.isclose(xx, 0.0)], [kpt_path[-1], ]))
    
    return kpath, bands, efermi, kpt_bounds, wkpts

# def get_bandInfo(in_file='band.xy'):
#     inp = open(in_file).readlines()
#     labels = inp[0].split()[1:]

#     # Data
#     BAND = np.array([line.split() for line in inp[1:] if line.strip()], dtype=float) 

#     kx = BAND[:, 0]
#     ky = BAND[:, 1]
#     kz = BAND[:, 2]
#     dist = BAND[:, 3]
#     eig = BAND[:, 4]
#     iband = BAND[:, 5]
#     line = BAND[:, 6]
    
#     kpath = dist
    
#     return kpath, bands, efermi, kpt_bounds, wkpts

__version__ = "0.0"

opts, args = command_line_arg()

kpath, bands, efermi, kpt_bounds, wkpts = get_bandInfo(opts.filename)
whts = None
# kx, ky, kz, dist, eig, iband, line, _ = get_bandInfo(opts.filename)

import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
# Use non-interactive backend in case there is no display
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['axes.unicode_minus'] = False

mpl_default_colors_cycle = [mpl.colors.to_hex(xx) for xx in
                            mpl.rcParams['axes.prop_cycle'].by_key()['color']]

bandplot(kpath, bands, efermi, kpt_bounds, opts, whts)

