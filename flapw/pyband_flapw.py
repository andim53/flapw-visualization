import os
import re
import numpy as np
from optparse import OptionParser

__version__ = "1.0"


def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option('-f', '--file',
                   action='store', type="string",
                   dest='filename', default='OUTCAR',
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


def get_bandInfo(inFile='OUTCAR'):
    """
    extract band energies from OUTCAR
    """

    outcar = [line for line in open(inFile) if line.strip()]

    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])

        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])

        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1

        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1
            # break

    # basis vector of reciprocal lattice
    # B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]],

    # When the supercell is too large, spaces are missing between real space
    # lattice constants. A bug found out by Wei Xie (weixie4@gmail.com).
    B = np.array([line.split()[-3:] for line in outcar[ibasis:ibasis+3]],
                 dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in outcar[Lvkpts:Lvkpts+nkpts]],
                   dtype=float)
    vkpts = tmp[:, :3]
    wkpts = tmp[:, -1]

    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2

    # in VASP 6.2, there is extra lines containing "Fermi energy: xxxx"
    if 'Fermi energy:' in outcar[LineEfermi]:
        N += ispin

    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'Fermi energy:' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        bands.append(float(line.split()[1]))

    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    if os.path.isfile('KPOINTS'):
        kp = open('KPOINTS').readlines()

    if os.path.isfile('KPOINTS') and kp[2][0].upper() == 'L':
        Nk_in_seg = int(kp[1].split()[0])
        Nseg = nkpts // Nk_in_seg
        vkpt_diff = np.zeros_like(vkpts, dtype=float)

        for ii in range(Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            vkpt_diff[start:end, :] = vkpts[start:end, :] - vkpts[start, :]

        kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
        # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
        for ii in range(1, Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            kpt_path[start:end] += kpt_path[start-1]

        # kpt_path /= kpt_path[-1]
        kpt_bounds = np.concatenate((kpt_path[0::Nk_in_seg], [kpt_path[-1], ]))
    else:
        # get band path
        vkpt_diff = np.diff(vkpts, axis=0)
        kpt_path = np.zeros(nkpts, dtype=float)
        kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
        # kpt_path /= kpt_path[-1]

        # get boundaries of band path
        xx = np.diff(kpt_path)
        kpt_bounds = np.concatenate(
            ([0.0, ], kpt_path[1:][np.isclose(xx, 0.0)], [kpt_path[-1], ]))

    return kpt_path, bands, Efermi, kpt_bounds, wkpts


opts, args = command_line_arg()

# kpath, bands, efermi, kpt_bounds, wkpts = get_bandInfo(opts.filename)

outcar = [line for line in open('OUTCAR') if line.strip()]

for ii, line in enumerate(outcar):
    if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])
            #print(f"nkpts = {nkpts} \nnband = {nband}")

    if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1

    if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

B = np.array([line.split()[-3:] for line in outcar[ibasis:ibasis+3]],
                dtype=float)

# print([line.split()[-3:] for line in outcar[ibasis:ibasis+3]]) 
# Results: [['0.313971743', '0.181271670', '0.000000000'], 
# ['0.000000000', '0.362543340', '0.000000000'], 
# ['0.000000000', '0.000000000', '0.028571429']]

# print(outcar[ibasis])
# Results: 3.185000000  0.000000000  0.000000000     0.313971743  0.181271670  0.000000000

# print(outcar[ibasis:ibasis+3])

