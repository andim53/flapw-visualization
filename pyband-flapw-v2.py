#!/usr/bin/env python

import pandas as pd

# Check if matplot GUI works, it will not works if you run script from a server
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except ImportError:
    mpl.use('agg')
from matplotlib.ticker import AutoMinorLocator

# Import regular expression, calling greek letter
import re

import numpy as np
from matplotlib.ticker import MultipleLocator
from optparse import OptionParser

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

def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version=__version__)

    par.add_option('-f', '--file',
                   action='store', type="string",
                   dest='filename', default='band.xy',
                   help='location of band.xy')
    
    par.add_option('-s', '--size', nargs=2,
                   action='store', type="float", dest='figsize',
                   default=(3.0, 4.0),
                   help='figure size of the output plot, ex: (3.0, 4.0)')
    
    par.add_option('-y', nargs=2,
                   action='store', type="float", dest='ylim',
                   default=None,
                   help='energy range of the band plot, ex: (-3,3)')
    
    par.add_option('--dpi',
                   action='store', type="int", dest='dpi',
                   default=360,
                   help='resolution of the output image, ex: 360')
    
    par.add_option('--lw',
                   action='store', type="float", dest='linewidth',
                   default=1.0,
                   help='linewidth of the band plot, ex: 1.0')
    
    par.add_option('-k', '--kpoints',
                   action='store', type="string", dest='kpts',
                   default=None,
                   help='kpoint path, use ",", " ", ";" or "-" to separate the K-point if necessary.')
    
    par.add_option('-o', '--output',
                   action='store', type="string", dest='bandimage',
                   default='band.png',
                   help='output image name, "band.png" by default')
    return par.parse_args()


__version__ = 1.0
opts, args = command_line_arg()

column_names = ['dist', 'eig', 'iband', 's', 'pz', 'px', 'py', 'dz2r2', 'dxz', 'dyz', 'dxy', 'dx2y2',
                'fm0', 'f+m1', 'f-m1', 'f+m2', 'f-m2', 'f+m3', 'f-m3']
data_bndatm = pd.read_csv(opts.filename, delim_whitespace=True, header=None, names=column_names, comment='#')

# Find the max line
max_line_value = data['line'].max()

x_positions_for_labels = []
for line_value in range(1, max_line_value + 1):
    first_occurrence = data[data['line'] == line_value]['dist'].iloc[0]
    x_positions_for_labels.append(first_occurrence)
    if line_value == max_line_value:
        last_occurrence = data[data['line'] == line_value]['dist'].iloc[-1]
        x_positions_for_labels.append(last_occurrence)

# Convert to array to get the min and max later
x_positions_for_labels = np.array(x_positions_for_labels)

# Initiazlize graph
width, height = opts.figsize
if opts.ylim:
    ymin, ymax = opts.ylim
else:
    _ = opts.ylim

dpi = opts.dpi

fig = plt.figure()
fig.set_size_inches(width, height)
ax = plt.subplot(111)

clrs = plt.cm.viridis(np.linspace(0, 1, len(data['iband'].unique())))

for iband, color in zip(data['iband'].unique(), clrs):
    subset = data[data['iband'] == iband]
    ax.plot(subset['dist'], 
            subset['eig'], 
            lw=opts.linewidth,
            linestyle='-', 
            color='b', 
            alpha=0.8, 
            zorder=0)
    
custom_labels = ['Γ', 'X', 'M', 'R', 'Z', 'A']

for pos in x_positions_for_labels:
    ax.axvline(x=pos, 
               color ='k', 
               linestyle='--', 
               linewidth=0.5,
               alpha=0.5)

ax.set_ylabel('Energy [eV]',
              labelpad=5)

ax.set_xticks(x_positions_for_labels)

if opts.kpts:
    ax.set_xticklabels(kpath_name_parse(opts.kpts))
else:
    ax.set_xticklabels([])

if opts.ylim:
    ax.set_ylim(ymin, ymax)

# Plot the min and max of X axis
ax.set_xlim(x_positions_for_labels.min(), x_positions_for_labels.max())

ax.set_xticks(x_positions_for_labels)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tight_layout(pad=0.20)
plt.savefig(opts.bandimage, dpi=opts.dpi)

# Diamond K-Path
# -k 'G, X, L, W, K, G' 

plt.show()