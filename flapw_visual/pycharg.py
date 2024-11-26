import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interactive, fixed  # Import fixed here

# グリッドサイズと電荷密度データの抽出、原子の座標取得
def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # グリッドサイズと電荷密度値の取得
    start_index = None
    for i, line in enumerate(content):
        if "BEGIN_DATAGRID_3D_charge_density" in line:
            start_index = i + 1
            break

    grid_size = list(map(int, content[start_index].strip().split()))
    nx, ny, nz = grid_size

    # 電荷密度値の抽出
    density_values = []
    for line in content[start_index + 1:]:
        if "END_DATAGRID_3D" in line:
            break
        density_values.extend(map(float, line.strip().split()))

    # データサイズ調整
    trimmed_density_values = density_values[:nx * ny * nz]
    density_array = np.array(trimmed_density_values).reshape((nx, ny, nz))

    # ATOMSセクションから原子の座標取得
    atom_positions = []
    atoms_section = False
    for line in content:
        if line.strip() == "ATOMS":
            atoms_section = True
            continue
        elif line.strip() == "BEGIN_BLOCK_DATAGRID_3D":
            break
        elif atoms_section:
            parts = line.split()
            if len(parts) >= 4:
                atom_positions.append(list(map(float, parts[1:4])))

    # 原子の座標をnumpy配列に変換
    atom_positions = np.array(atom_positions)

    return density_array, atom_positions, nx, ny, nz

def plot_3d(density_array, atom_positions, nx, ny, nz, angle_x=30, angle_y=30, threshold=0.1):
    """
    Plot a 3D charge density surface with atomic positions.

    Parameters:
    - density_array: 3D numpy array of charge density values
    - atom_positions: 2D numpy array of atomic positions (shape: n_atoms x 3)
    - nx, ny, nz: Dimensions of the 3D grid
    - angle_x, angle_y: Angles for 3D plot view
    - threshold: Contour level for the marching cubes algorithm (default 0.1)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 電荷密度の等値面プロット (しきい値を0.1に設定)
    verts, faces, _, _ = measure.marching_cubes(density_array, level=threshold)

    # Triangular surface plot for charge density
    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           cmap='Blues', alpha=0.5, edgecolor='none', linewidth=0)

    # Add colorbar for the mesh
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)
    colorbar.set_label('Charge Density', fontsize=12)

    # 原子位置のプロット (Normalize atom positions to grid size)
    ax.scatter(atom_positions[:, 0] / 6.74 * (nx - 1), 
               atom_positions[:, 1] / 6.74 * (ny - 1), 
               atom_positions[:, 2] / 6.74 * (nz - 1),
               color='red', s=80, label='Atoms', marker='o', edgecolor='black', linewidth=1)

    # グラフの装飾
    ax.set_title("3D Charge Density with Atomic Positions", fontsize=16, weight='bold')
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_zlabel("Z-axis", fontsize=12)
    ax.view_init(angle_x, angle_y)
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()


# インタラクティブなプロット設定
def interactive_plot(file_path):
    """
    Set up an interactive 3D plot with adjustable viewing angles for the charge density and atomic positions.

    Parameters:
    - file_path: Path to the data file containing the charge density and atomic positions.
    """
    # データの読み込みと処理
    density_array, atom_positions, nx, ny, nz = load_and_process_data(file_path)
    
    # インタラクティブな視点設定 (angle_x and angle_yを調整可能)
    interactive_plotter = interactive(
        plot_3d, 
        density_array=fixed(density_array),
        atom_positions=fixed(atom_positions),
        nx=fixed(nx), ny=fixed(ny), nz=fixed(nz),
        angle_x=(-90, 90, 10),  # x軸の角度調整範囲 (-90度から90度)
        angle_y=(-90, 90, 10)   # y軸の角度調整範囲 (-90度から90度)
    )
    
    # インタラクティブなプロットを表示
    display(interactive_plotter)

# 実行: ファイルパスを指定してインタラクティブプロットを表示
file_path = 'input/den.xsf'
interactive_plot(file_path)
