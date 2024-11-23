import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def get_options(
    dos='dos.xy', figsize=(4.8, 3.0), elim=None, dpi=360, ylabel="Density of States (DOS)",
    xlabel="Energy (eV)", mpl_style='classic', zero=0.0, legendloc='lower right', 
    vertical=False, dosimage='dos.png'
):
    """
    Return a dictionary of options for the DOS plot.

    Parameters:
    -----------
    dos : str
        Path to the DOS data file.
    figsize : tuple of float
        Size of the plot figure (width, height) in inches.
    elim : tuple of float, optional
        Limits for the energy axis (min, max). Defaults to None (auto-scaling).
    dpi : int
        Dots per inch for the output image resolution. Defaults to 360.
    ylabel : str
        Label for the y-axis (horizontal mode) or x-axis (vertical mode).
    xlabel : str
        Label for the x-axis (horizontal mode) or y-axis (vertical mode).
    mpl_style : str
        Matplotlib style to apply. Defaults to 'classic'.
    zero : float
        Energy reference level for the plot. Defaults to 0.0.
    legendloc : str
        Location of the legend in the plot. Defaults to 'lower right'.
    vertical : bool
        If True, the plot orientation is vertical. Defaults to False.
    dosimage : str
        Name of the output image file. Defaults to 'dos.png'.

    Returns:
    --------
    dict
        Dictionary containing all options.
    """
    return {
        'dos': dos,
        'figsize': figsize,
        'elim': elim,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'dpi': dpi,
        'mpl_style': mpl_style,
        'zero': zero,
        'legendloc': legendloc,
        'vertical': vertical,
        'dosimage': dosimage,
    }

def readDOSFromFile(opts):
    """
    Read and process the DOS data from a file.

    The DOS file format:
    - First line: column labels (e.g., # eng dos occ e_sum).
    - Subsequent lines: energy, total DOS, occupation, and energy sum.

    Parameters:
    -----------
    opts : dict
        Dictionary containing the DOS file path under the key 'dos'.

    Returns:
    --------
    tuple
        Arrays for energy (`xen`), total DOS (`tdos`), occupation (`occ`), 
        and energy sum (`e_sum`).
    """
    try:
        with open(opts['dos'], 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"The DOS file '{opts['dos']}' does not exist.")
    except IOError as e:
        raise IOError(f"Error reading the file '{opts['dos']}': {e}")

    # Parse labels from the first line
    labels = lines[0].strip().split()[1:]

    # Parse data into a numpy array
    data = np.array([list(map(float, line.split())) for line in lines[1:] if line.strip()])

    xen, tdos, occ, e_sum = data.T  # Transpose into separate arrays
    opts['pdosLabel'] = labels  # Store labels in the options dictionary

    return xen, tdos, occ, e_sum

def dosplot(xen, tdos, opts):
    """
    Plot the Density of States (DOS) with configurable options.

    Parameters:
    -----------
    xen : numpy.ndarray
        Energy values (x-axis in horizontal mode or y-axis in vertical mode).
    tdos : numpy.ndarray
        Total DOS values.
    opts : dict
        Dictionary containing plot options.
    """
    width, height = opts['figsize']

    # Apply matplotlib style
    plt.style.use(opts['mpl_style'])
    mpl.rcParams['axes.unicode_minus'] = False

    # Adjust energy reference
    xen -= opts['zero']

    # Create the figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=opts['dpi'])

    # Plot orientation
    if opts['vertical']:
        # Vertical orientation
        ax.fill_betweenx(xen, tdos, color='black', alpha=0.2, zorder=1)
        line, = ax.plot(tdos, xen, color='black', lw=1.5, zorder=2)
        ax.set_xlabel(opts['ylabel'], fontsize=14, labelpad=10)
        ax.set_ylabel(opts['xlabel'], fontsize=14, labelpad=10)
        if opts['elim']:
            ax.set_ylim(opts['elim'])
    else:
        # Horizontal orientation
        ax.fill_between(xen, tdos, color='black', alpha=0.2, zorder=1)
        line, = ax.plot(xen, tdos, color='black', lw=1.5, zorder=2)
        ax.set_xlabel(opts['xlabel'], fontsize=14, labelpad=10)
        ax.set_ylabel(opts['ylabel'], fontsize=14, labelpad=10)
        if opts['elim']:
            ax.set_xlim(opts['elim'])

    # Configure ticks and grid
    ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.2)
    ax.tick_params(axis='both', which='minor', length=5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Add legend
    ax.legend([line], ['Total DOS'], loc=opts['legendloc'], fontsize=12, frameon=True, framealpha=0.7)

    # Save and show the plot
    plt.tight_layout(pad=1.2)
    plt.savefig(opts['dosimage'], dpi=opts['dpi'], bbox_inches='tight')
    plt.show()

'''
# Example usage:
import flapw_visual.pydos as pdos

input_file = "./input/dos.xy"
output_file = "./output/dos.png"

# Define the options (these can be passed from another script or interactively)
opts = pdos.get_options(dos=input_file, figsize=(6, 4), elim=None, dpi=300,
                      mpl_style='classic', zero=0.0, vertical=True,
                      legendloc='upper right', dosimage=output_file,
                      )

# Read the DOS data
xen, tdos, occ, e_sum = pdos.readDOSFromFile(opts)

pdos.dosplot(xen, tdos, opts) 
'''
