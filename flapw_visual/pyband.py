#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import re
import numpy as np

# Greek Letters Dictionary for k-points
GREEK_KPTS = {
    'G': r'$\mathrm{\mathsf{\Gamma}}$',
    'GAMMA': r'$\mathrm{\mathsf{\Gamma}}$',
    'DELTA': r'$\mathrm{\mathsf{\Delta}}$',
    'LAMBDA': r'$\mathrm{\mathsf{\Lambda}}$',
    'SIGMA': r'$\mathrm{\mathsf{\Sigma}}$',
}

def kpath_name_parse(KPATH_STR):
    """
    Parse the kpath string to handle Greek characters and generalize.
    """
    KPATH_STR = KPATH_STR.upper()
    KPATH_SEPARATORS = ' ,-;'

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

def get_option(
    band='band.xy', figsize=(4.0, 5.0), xlim=None, elim=None, ylabel="Energy (eV)", 
    xlabel="Wave Vector", dpi=360, kpts="G, X, U|K, R, Z, A", linewidth=0.5, 
    bandimage='band.png'
):
    """
    Retrieve options for plotting the band structure.

    Parameters:
    -----------
    (Descriptions of parameters unchanged...)

    Returns:
    --------
    dict
        Dictionary of parsed options.
    """
    return {
        'filename': band,
        'figsize': figsize,
        'elim': elim,
        'xlim': xlim,
        'dpi': dpi,
        'kpts': kpts,
        'ylabel': ylabel,
        'xlabel': xlabel,
        'bandimage': bandimage,
        'linewidth': linewidth
    }

def readBANDFromFile(opts):
    """
    Read band structure data from a file and extract k-point positions.

    Parameters:
    -----------
    opts : dict
        Dictionary of options containing the file path under 'filename'.

    Returns:
    --------
    tuple
        - pandas.DataFrame containing band structure data.
        - numpy.ndarray of x-axis positions for k-point labels.
    """
    try:
        data = pd.read_csv(
            opts['filename'], sep=r'\s+', skiprows=1, 
            names=['kx', 'ky', 'kz', 'dist', 'eig', 'iband', 'line']
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"File {opts['filename']} not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File {opts['filename']} is empty or malformed.")
    
    max_line = data['line'].max()
    x_positions = [
        data[data['line'] == i]['dist'].iloc[0] for i in range(1, max_line + 1)
    ]
    x_positions.append(data[data['line'] == max_line]['dist'].iloc[-1])

    return data, np.array(x_positions)

def band_plot(data, opts, x_positions):
    """
    Plot the electronic band structure.

    Parameters:
    -----------
    data : pandas.DataFrame
        Band structure data.
    opts : dict
        Plotting options.
    x_positions : numpy.ndarray
        Positions for x-axis ticks (k-point labels).
    """
    fig, ax = plt.subplots(figsize=opts['figsize'], dpi=opts['dpi'])
    
    # Plot each band
    for iband in data['iband'].unique():
        subset = data[data['iband'] == iband]
        ax.plot(subset['dist'], subset['eig'], lw=opts['linewidth'], 
                color='black', alpha=0.7, zorder=2)
    
    # Add vertical lines at k-point positions
    for pos in x_positions:
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # Set axis labels
    ax.set_ylabel(opts['ylabel'], fontsize=14, labelpad=10)
    ax.set_xlabel(opts['xlabel'], fontsize=14, labelpad=10)
    
    # Set x-ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(kpath_name_parse(opts['kpts']), fontsize=12)

    # Apply y and x-axis limits
    if opts['elim']:
        ax.set_ylim(opts['elim'])
    ax.set_xlim(x_positions.min(), x_positions.max() if opts['xlim'] is None else opts['xlim'])
    
    # Grid and minor ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.2, alpha=0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Save and display the plot
    plt.tight_layout(pad=1.0)
    plt.savefig(opts['bandimage'], dpi=opts['dpi'])
    plt.show()

'''
# Example usage:

import flapw_visual.pyband as pb

input_file = "./input/band.xy"
output_file = "./output/band.png"

opts = pb.get_option(band=input_file, figsize=(4.0, 5.0), elim=None, ylabel="Energy (eV)", xlabel="Wave Vector",
            dpi=360, kpts="G, X, U|K, R, Z, A", linewidth=0.5, bandimage=output_file)

data, x_positions_for_labels = pb.readBANDFromFile(opts)

# Call the band plotting function
pb.band_plot(data, opts, x_positions_for_labels)
'''
