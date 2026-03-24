import numpy as np
import matplotlib.pyplot as plt
import os
from LumAPI import AngularSpectrum_Vector

def plot_diff(diff_data, title, xlabel, ylabel, extent, filename, is_xy=True, vmax=1e-10):
    plt.figure(figsize=(5, 4) if is_xy else (10, 2.5))
    im = plt.imshow(diff_data.T if is_xy else diff_data, 
                    extent=extent, cmap='viridis', origin='lower', aspect='auto',
                    vmin=0, vmax=vmax)
    
    max_err = np.max(diff_data)
    plt.title(f"{title}\nMax Diff: {max_err:.2e}")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(filename, dpi=150 if is_xy else 200)
    plt.close()

def run_as_vector_validation():
    os.makedirs('pics', exist_ok=True)
    
    # === 物理参数设置 ===
    lamb = 1.55e-6         
    k = 2 * np.pi / lamb
    D = 60e-6              
    dx = 0.5e-6            
    f_design = 50e-6       
    
    x_n = np.arange(-D/2, D/2, dx)
    y_n = np.arange(-D/2, D/2, dx)
    X_n, Y_n = np.meshgrid(x_n, y_n, indexing='xy')
    aperture = (X_n**2 + Y_n**2) <= (D/2)**2
    
    Ny, Nx = len(y_n), len(x_n)
    z_scan = np.linspace(1e-6, 100e-6, 200) 
    
    field_results = {}
    
    # ----------------------------------------------------
    # 阶段一：正负约定与 FFT / Numba 模式比对
    # ----------------------------------------------------
    for software in ['+', '-']:
        print(f"\n========== 开始计算: 约定={software} ==========")
        fn_sg = 'plus' if software == '+' else 'minus'
        sg = 1.0 if software == '+' else -1.0
        
        phase = -sg * k * np.sqrt(X_n**2 + Y_n**2 + f_design**2)
        E_near_x = aperture * np.exp(1j * phase)
        E_near_y = np.zeros_like(E_near_x)
        
        # ===================================================
        # 1. FFT 模式全空间一次性求解
        # ===================================================
        print("  -> 运行 FFT 模式 (极速3D计算)...")
        E_tot_f, _, _, _ = AngularSpectrum_Vector(lamb, x_n, y_n, E_near_x, E_near_y, x_n, y_n, z_scan, mode='f', software=software)
        
        # (1) Z 轴扫描 (找到真实焦平面 actual_f)
        I_z_axis_f = E_tot_f[Ny//2, Nx//2, :]**2
        actual_f_idx_f = np.argmax(I_z_axis_f)
        actual_f_f = z_scan[actual_f_idx_f]
        
        plt.figure(figsize=(5, 4))
        plt.plot(z_scan*1e6, I_z_axis_f, 'b-', linewidth=2)
        plt.axvline(f_design*1e6, color='r', linestyle='--', label=f'Design ({f_design*1e6} μm)')
        plt.axvline(actual_f_f*1e6, color='g', linestyle=':', label=f'Actual ({actual_f_f*1e6:.1f} μm)')
        plt.title(f"Z-axis Vector Intensity ({software} | fft)")
        plt.xlabel("Z (μm)"); plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_fft_Z.jpg', dpi=150)
        plt.close()

        # (2) XY 焦平面扫描
        I_xy_f = E_tot_f[:, :, actual_f_idx_f]**2   # FFT返回值形状: (Ny, Nx)
        
        plt.figure(figsize=(5, 4))
        plt.imshow(I_xy_f, extent=[x_n[0]*1e6, x_n[-1]*1e6, y_n[0]*1e6, y_n[-1]*1e6], cmap='hot', origin='lower')
        plt.title(f"Focal Plane XY ({software} | fft)")
        plt.xlabel("X (μm)"); plt.ylabel("Y (μm)")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_fft_XY.jpg', dpi=150); plt.close()
        
        # (3) XZ 全景扫描
        I_xz_f = E_tot_f[Ny//2, :, :]**2          # FFT返回值形状: (Nx, Nz)

        plt.figure(figsize=(10, 2.5))
        plt.imshow(I_xz_f, extent=[z_scan[0]*1e6, z_scan[-1]*1e6, x_n[0]*1e6, x_n[-1]*1e6], cmap='jet', aspect='auto', origin='lower')
        plt.axvline(f_design*1e6, color='w', linestyle='--', alpha=0.5)
        plt.title(f"XZ Propagation Plane ({software} | fft)")
        plt.xlabel("Z (μm)"); plt.ylabel("X (μm)")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_fft_XZ.jpg', dpi=200); plt.close()

        # 缓存 FFT 结果
        field_results[(software, 'fft')] = {'I_xy': I_xy_f, 'I_xz': I_xz_f}


        # ===================================================
        # 2. Numba 模式针对性切片计算
        # ===================================================
        print("  -> 运行 Numba 模式 (精确坐标反演积分)...")
        # (1) Z 轴扫描 
        E_tot_n_z, _, _, _ = AngularSpectrum_Vector(lamb, x_n, y_n, E_near_x, E_near_y, [0.0], [0.0], z_scan, mode='n', software=software)
        I_z_axis_n = E_tot_n_z[0, 0, :]**2
        actual_f_n = z_scan[np.argmax(I_z_axis_n)]

        plt.figure(figsize=(5, 4))
        plt.plot(z_scan*1e6, I_z_axis_n, 'b-', linewidth=2)
        plt.axvline(f_design*1e6, color='r', linestyle='--', label=f'Design ({f_design*1e6} μm)')
        plt.axvline(actual_f_n*1e6, color='g', linestyle=':', label=f'Actual ({actual_f_n*1e6:.1f} μm)')
        plt.title(f"Z-axis Vector Intensity ({software} | numba)")
        plt.xlabel("Z (μm)"); plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_numba_Z.jpg', dpi=150)
        plt.close()

        # (2) XY 焦平面扫描
        # 注意: x_far=x_n, y_far=y_n 返回的网格顺序为 (Nx, Ny)
        E_tot_n_xy, _, _, _ = AngularSpectrum_Vector(lamb, x_n, y_n, E_near_x, E_near_y, x_n, y_n, [actual_f_n], mode='n', software=software)
        I_xy_n = E_tot_n_xy[:, :, 0]**2 

        plt.figure(figsize=(5, 4))
        # Numba数据转置匹配绘图维度
        plt.imshow(I_xy_n.T, extent=[x_n[0]*1e6, x_n[-1]*1e6, y_n[0]*1e6, y_n[-1]*1e6], cmap='hot', origin='lower')
        plt.title(f"Focal Plane XY ({software} | numba)")
        plt.xlabel("X (μm)"); plt.ylabel("Y (μm)")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_numba_XY.jpg', dpi=150); plt.close()
        
        # (3) XZ 全景扫描
        E_tot_n_xz, _, _, _ = AngularSpectrum_Vector(lamb, x_n, y_n, E_near_x, E_near_y, x_n, [0.0], z_scan, mode='n', software=software)
        I_xz_n = E_tot_n_xz[:, 0, :]**2 # (Nx, Nz)

        plt.figure(figsize=(10, 2.5))
        plt.imshow(I_xz_n, extent=[z_scan[0]*1e6, z_scan[-1]*1e6, x_n[0]*1e6, x_n[-1]*1e6], cmap='jet', aspect='auto', origin='lower')
        plt.axvline(f_design*1e6, color='w', linestyle='--', alpha=0.5)
        plt.title(f"XZ Propagation Plane ({software} | numba)")
        plt.xlabel("Z (μm)"); plt.ylabel("X (μm)")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(f'pics/AS_vector_{fn_sg}_numba_XZ.jpg', dpi=200); plt.close()
        
        field_results[(software, 'numba')] = {'I_xy': I_xy_n, 'I_xz': I_xz_n}

    # ----------------------------------------------------
    # 阶段二：误差对比图
    # ----------------------------------------------------
    print("\n========== 开始生成误差对比图 ==========")
    xy_extent = [x_n[0]*1e6, x_n[-1]*1e6, y_n[0]*1e6, y_n[-1]*1e6]
    xz_extent = [z_scan[0]*1e6, z_scan[-1]*1e6, x_n[0]*1e6, x_n[-1]*1e6]
    diff_vmax = 1e-10

    # 1. 模式差值: Numba 与 FFT (注意 I_xy_n 需要转置以匹配 FFT 的 Ny,Nx 维度)
    diff_xy_mode = np.abs(field_results[('+', 'numba')]['I_xy'].T - field_results[('+', 'fft')]['I_xy'])
    diff_xz_mode = np.abs(field_results[('+', 'numba')]['I_xz'] - field_results[('+', 'fft')]['I_xz'])
    
    print(f"[Numba - FFT 模式误差]:")
    print(f"  XY: Max = {np.max(diff_xy_mode):.4e}, Mean = {np.mean(diff_xy_mode):.4e}")
    print(f"  XZ: Max = {np.max(diff_xz_mode):.4e}, Mean = {np.mean(diff_xz_mode):.4e}")
    
    plot_diff(diff_xy_mode, "Diff: Numba - FFT (XY)", "X (μm)", "Y (μm)", xy_extent, 'pics/AS_vector_diff_numba_fft_XY.jpg', is_xy=True, vmax=diff_vmax)
    plot_diff(diff_xz_mode, "Diff: Numba - FFT (XZ)", "Z (μm)", "X (μm)", xz_extent, 'pics/AS_vector_diff_numba_fft_XZ.jpg', is_xy=False, vmax=diff_vmax)

    # 2. 约定差值: Minus vs Plus (同在 FFT 模式下对比)
    diff_xy_conv = np.abs(field_results[('-', 'fft')]['I_xy'] - field_results[('+', 'fft')]['I_xy'])
    diff_xz_conv = np.abs(field_results[('-', 'fft')]['I_xz'] - field_results[('+', 'fft')]['I_xz'])
    
    print(f"[Minus - Plus 约定误差]:")
    print(f"  XY: Max = {np.max(diff_xy_conv):.4e}, Mean = {np.mean(diff_xy_conv):.4e}")
    print(f"  XZ: Max = {np.max(diff_xz_conv):.4e}, Mean = {np.mean(diff_xz_conv):.4e}")

    plot_diff(diff_xy_conv, "Diff: Minus - Plus Phase Conv (XY)", "X (μm)", "Y (μm)", xy_extent, 'pics/AS_vector_diff_minus_plus_XY.jpg', is_xy=True, vmax=diff_vmax)
    plot_diff(diff_xz_conv, "Diff: Minus - Plus Phase Conv (XZ)", "Z (μm)", "X (μm)", xz_extent, 'pics/AS_vector_diff_minus_plus_XZ.jpg', is_xy=False, vmax=diff_vmax)

def run_as_feature_analysis():
    """使用 Numba 模式实现焦点区域的微米级无级高分辨率变焦"""
    print("\n========== 开始矢量角谱特性分析 (Numba 无级缩放) ==========")
    lamb = 1.55e-6         
    k = 2 * np.pi / lamb
    D = 60e-6              
    dx = 0.5e-6            
    f_design = 50e-6       
    
    x_n = np.arange(-D/2, D/2, dx)
    y_n = np.arange(-D/2, D/2, dx)
    X_n, Y_n = np.meshgrid(x_n, y_n, indexing='xy')
    aperture = (X_n**2 + Y_n**2) <= (D/2)**2
    
    phase = -k * np.sqrt(X_n**2 + Y_n**2 + f_design**2)
    E_near_x = aperture * np.exp(1j * phase)
    E_near_y = np.zeros_like(E_near_x)
    
    # 快速扫描以找到高分辨作图所需的准确焦距 (Z)
    z_scan_fast = np.linspace(40e-6, 60e-6, 100)
    E_tot_z, _, _, _ = AngularSpectrum_Vector(lamb, x_n, y_n, E_near_x, E_near_y, [0.0], [0.0], z_scan_fast, mode='n', software='+')
    actual_f = z_scan_fast[np.argmax(E_tot_z[0, 0, :]**2)]

    # 在焦平面上进行局部高分辨率扫描 (提取核心 ±4 μm)
    x_f_zoom = np.linspace(-4e-6, 4e-6, 120)
    
    E_tot, Ex, _, Ez = AngularSpectrum_Vector(
        lamb, x_n, y_n, E_near_x, E_near_y, 
        x_f_zoom, x_f_zoom, [actual_f], 
        mode='n', software='+'
    )
    
    Ix = np.abs(Ex[:, :, 0])**2
    Iz = np.abs(Ez[:, :, 0])**2
    Itot = E_tot[:, :, 0]**2

    def plot_component(data, title, filename, cmap='hot'):
        plt.figure(figsize=(5, 4))
        im = plt.imshow(data.T, extent=[-4, 4, -4, 4], cmap=cmap, origin='lower')
        plt.title(f"{title}\nMax: {np.max(data):.2e}")
        plt.xlabel("X (μm)"); plt.ylabel("Y (μm)")
        plt.colorbar(im); plt.tight_layout()
        plt.savefig(filename, dpi=150); plt.close()

    plot_component(Ix, "Intensity |Ex|² (Main Component)", 'pics/AS_vector_feature_Ex.jpg')
    plot_component(Iz, "Intensity |Ez|² (Longitudinal)", 'pics/AS_vector_feature_Ez.jpg', cmap='plasma')
    plot_component(Itot, "Total Intensity |E_total|²", 'pics/AS_vector_feature_Total.jpg')
    
    print("无级缩放高分辨特征图已生成！")

if __name__ == "__main__":
    print("1: 运行角谱法全量模式验证与误差分析 (FFT vs Numba)")
    run_as_vector_validation()
    print("2: 仅运行局部焦点无级变焦分析 (高 NA 偏振耦合展现)")
    run_as_feature_analysis()