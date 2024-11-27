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