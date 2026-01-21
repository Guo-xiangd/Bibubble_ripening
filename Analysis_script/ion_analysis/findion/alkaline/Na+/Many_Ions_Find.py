#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from tqdm import tqdm
import warnings

# 修复警告过滤问题
warnings.filterwarnings("ignore", category=UserWarning)  # 过滤用户警告
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 过滤弃用警告

def setup_output_files(save_path, scheme, step, cutoff, trj_name, findpart, ion_num_upper):
    """创建输出文件并写入表头"""
    # 设置文件名
    index_file = os.path.join(save_path, f"{scheme}_ohidx_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    h_xyz_file = os.path.join(save_path, f"{scheme}_h_xyz_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    o_xyz_file = os.path.join(save_path, f"{scheme}_o_xyz_dump{step}_cut{cutoff}_{trj_name}_{findpart}.txt")
    
    # 创建并写入文件头
    with open(index_file, 'w') as f_idx, open(h_xyz_file, 'w') as f_h, open(o_xyz_file, 'w') as f_o:
        # 写入索引文件头
        f_idx.write(f"# {'Ions':>6}{'MD_step':>16}")
        f_h.write(f"# {'Ions':>6}{'MD_step':>16}")
        f_o.write(f"# {'Ions':>6}{'MD_step':>16}")
        
        
        # 根据方案写入表头
        if scheme == 'h3o':
            for i in range(ion_num_upper):
                f_idx.write(f"{f'O{i+1}':>8}")
                f_h.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{f'O{i+1}':>8}")
                for j in range(3):
                    f_idx.write(f"{f'H{j+1}':>8}")
                    f_h.write(f"{f'H{j+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{'X':>16}{'Y':>16}{'Z':>16}")
        else:  # 'oh'
            for i in range(ion_num_upper):
                f_idx.write(f"{f'O{i+1}':>8}{'H':>8}")
                f_h.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}{'H':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                f_o.write(f"{f'O{i+1}':>8}{'X':>16}{'Y':>16}{'Z':>16}")
                
        f_idx.write("\n")
        f_h.write("\n")
        f_o.write("\n")        
    
    # 返回文件对象用于追加写入
    return open(index_file, 'a'), open(h_xyz_file, 'a'), open(o_xyz_file, 'a')

def extract_box_from_timestep(u, frame_idx):
    """
    直接从轨迹文件中提取当前帧的盒子信息
    返回格式: [Lx, Ly, Lz, 90.0, 90.0, 90.0] (适用于正交盒子)
    """
    u.trajectory[frame_idx]  # 定位到指定帧
    ts = u.trajectory.ts
    
    # 尝试从底层数据中提取盒子信息
    try:
        # 获取原始单元信息
        unitcell = ts._unitcell
        #print(unitcell) unitcell lx still keeps 2L for [-L,L]
        
        dimensions = unitcell
        return dimensions
    except Exception as e:
        # 如果无法获取盒子信息，尝试从原子位置推导
        try:
            print(f"警告: 无法从轨迹获取盒子信息，尝试从原子位置推导: {str(e)}")
            
            # 获取所有原子位置
            coords = u.atoms.positions
            
            # 如果轨迹为空，返回None
            if len(coords) == 0:
                return None
                
            # 计算边界框
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            # 计算尺寸
            Lx = max_coords[0] - min_coords[0]
            Ly = max_coords[1] - min_coords[1]
            Lz = max_coords[2] - min_coords[2]
            
            # 构建标准盒子信息
            dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
            return dimensions
        except Exception as e2:
            print(f"无法获取盒子信息: {str(e2)}")
            return None

        

def find_ions(scheme, system, cutoff, u, frame_idx):
    """在当前时间步中查找离子"""
    use_pbc = (system != 'sphere')
    
    # 手动提取盒子信息 - 格式为 [Lx, Ly, Lz, 90, 90, 90]
    dimensions = extract_box_from_timestep(u, frame_idx)
    #print(dimensions)
    
    oxygens = u.select_atoms('type 2')  # O原子
    hydrogens = u.select_atoms('type 1')  # H原子
    
    # 如果没有氧原子或氢原子，返回空结果
    if len(oxygens) == 0 or len(hydrogens) == 0:
        return {-1: []}, {-1: []}, {-1: []}
    
    # 获取键合信息
    o_h_pairs = []
    
    try:
        # 计算原子间距离 - 使用六元素盒子尺寸
        pairs = capped_distance(
            oxygens.positions, 
            hydrogens.positions,
            max_cutoff=cutoff, 
            box=dimensions,  # 现在使用正确的格式
            return_distances=False
        )
        
        # 处理键对
        for o_idx, h_idx in pairs:
            o_id = oxygens[o_idx].id
            h_id = hydrogens[h_idx].id
            h_pos = hydrogens[h_idx].position
            o_h_pairs.append((o_id, h_id, h_pos))
    
    except Exception as e:
        print(f"计算距离时出错: {str(e)}")
        return {-1: []}, {-1: []}, {-1: []}
    
    # 按氧原子分组
    o_h_bonds = {}
    for o_id, h_id, h_pos in o_h_pairs:
        if o_id not in o_h_bonds:
            o_h_bonds[o_id] = []
        o_h_bonds[o_id].append((h_id, h_pos))
    
    # 根据方案分类离子
    o_dict, h_coord_dict, o_coord_dict = {}, {}, {}
    for o_id, h_list in o_h_bonds.items():
        o_pos = oxygens.select_atoms(f'id {o_id}').positions[0]
        
        if scheme == 'h3o' and len(h_list) == 3:  # 水合氢离子
            o_dict[o_id] = [h[0] for h in h_list]
            h_coord_dict[o_id] = [h[1] for h in h_list]
            o_coord_dict[o_id] = o_pos
        elif scheme == 'oh' and len(h_list) == 1:  # 氢氧根离子
            o_dict[o_id] = [h[0] for h in h_list]
            h_coord_dict[o_id] = [h[1] for h in h_list]
            o_coord_dict[o_id] = o_pos
    
    # 如果没有找到离子
    if not o_dict:
        o_dict = {-1: []}
        o_coord_dict = {-1: []}
        h_coord_dict = {-1: []}
    
    return o_dict, o_coord_dict, h_coord_dict

def write_placeholder(f_idx, f_h, f_o, scheme):
    """写入占位符数据"""
    if scheme == 'h3o':
        f_idx.write(f"{'-1':>8}{'-1':>8}{'-1':>8}{'-1':>8}")
        f_h.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}")
        for _ in range(3):  # 3个H原子
            f_h.write(f"{0:>8}{'0':>16}{'0':>16}{'0':>16}")
    else:  # 'oh'
        f_idx.write(f"{'-1':>8}{'-1':>8}")
        f_h.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}{0:>8}{'0':>16}{'0':>16}{'0':>16}")
    
    f_o.write(f"{'-1':>8}{'0':>16}{'0':>16}{'0':>16}")

def write_ion_data(f_idx, f_h, f_o, o_id, o_pos, h_ids, h_positions, scheme, dimensions):
    """写入离子数据"""
    
    # 计算平移量
    shift_x = -dimensions[0] / 2  # x方向减去L
    shift_y = -dimensions[1] / 2  # y方向减去0.5L
    shift_z = -dimensions[2] / 2  # z方向减去0.5L    
 
    # 平移氧原子坐标
    shifted_o_pos = [
        o_pos[0] + shift_x,
        o_pos[1] + shift_y,
        o_pos[2] + shift_z
    ]
    
    # 平移氢原子坐标
    shifted_h_positions = [
        [
            pos[0] + shift_x,
            pos[1] + shift_y,
            pos[2] + shift_z
        ] for pos in h_positions
    ]
    
    # 写入索引文件
    f_idx.write(f"{o_id:>8}")
    if scheme == 'h3o':
        for h_id in h_ids:
            f_idx.write(f"{h_id:>8}")
    else:  # 'oh'
        f_idx.write(f"{h_ids[0]:>8}" if h_ids else "-1")  # 只写入第一个H原子ID
    
    # 写入坐标文件
    f_h.write(f"{o_id:>8}{shifted_o_pos[0]:>16.8f}{shifted_o_pos[1]:>16.8f}{shifted_o_pos[2]:>16.8f}")
    for i, h_pos in enumerate(shifted_h_positions):
        h_id = h_ids[i] if i < len(h_ids) else -1
        f_h.write(f"{h_id:>8}{h_pos[0]:>16.8f}{h_pos[1]:>16.8f}{h_pos[2]:>16.8f}")
    
    # 写入氧坐标
    f_o.write(f"{o_id:>8}{shifted_o_pos[0]:>16.8f}{shifted_o_pos[1]:>16.8f}{shifted_o_pos[2]:>16.8f}")

if __name__ == '__main__':
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='离子检测程序')
    
    # 添加命令行参数
    parser.add_argument('trj_path', type=str, help='轨迹文件路径')
    parser.add_argument('save_path', type=str, help='保存路径')
    parser.add_argument('trj_file', type=str, help='轨迹文件名')
    parser.add_argument('--findpart', type=int, default=1, help='要处理的部分 (默认: 1)')
    parser.add_argument('--choose_part', type=int, default=2, help='总分割数 (默认: 2)')
    parser.add_argument('--step', type=int, default=1, help='帧步长 (默认: 1)')
    parser.add_argument('--scheme', type=str, default='h3o', choices=['oh', 'h3o'], help='检测方案: oh 或 h3o (默认: h3o)')
    parser.add_argument('--system', type=str, default='bulk', choices=['slab', 'sphere', 'bulk'], help='系统类型 (默认: bulk)')
    parser.add_argument('--ion_num', type=int, default=40, help='预期离子数 (默认: 40)')
    parser.add_argument('--cutoff', type=float, default=1.20, help='截断距离 (默认: 1.20)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置参数
    trj_path = args.trj_path
    save_path = args.save_path
    trj_file = args.trj_file
    findpart = args.findpart
    choose_part = args.choose_part
    step = args.step
    scheme = args.scheme
    system = args.system
    ion_num = args.ion_num
    ion_num_upper = ion_num * 2 + 2
    cutoff = args.cutoff
    
    trj_name = trj_file.split('.')[0]
    
    print(f"开始离子检测: {trj_file}")
    print(f"方案: {scheme}, 系统: {system}, 截断距离: {cutoff}Å")
    print(f"处理部分: {findpart}/{choose_part}")
    
    # 加载轨迹文件
    try:
        full_path = os.path.join(trj_path, trj_file)
        u = mda.Universe(full_path, format='LAMMPSDUMP')
        total_frames = len(u.trajectory)
        frames_per_part = total_frames // choose_part
        start_frame = findpart * frames_per_part
        stop_frame = min((findpart + 1) * frames_per_part, total_frames)
        
        print(f"加载的轨迹有 {total_frames} 帧")
        print(f"处理范围: 帧 {start_frame} 到 {stop_frame} (共 {stop_frame - start_frame} 帧)")
        print(f"最大离子数: {ion_num_upper}")
        
        # 准备输出文件
        f_idx, f_h, f_o = setup_output_files(save_path, scheme, step, cutoff, trj_name, findpart, ion_num_upper)
        
        # 处理轨迹
        frame_count = 0
        with tqdm(total=(stop_frame - start_frame), desc="处理帧进度") as pbar:
            for frame_idx in range(start_frame, stop_frame, step):
                frame_count += 1
                try:
                    # 定位到当前帧
                    u.trajectory[frame_idx]
                    md_step = u.trajectory.frame
                    
                    # 获取盒子尺寸
                    dimensions = extract_box_from_timestep(u, frame_idx)
                    
                    # 查找离子
                    o_dict, o_coord_dict, h_coord_dict = find_ions(
                        scheme, system, cutoff, u, frame_idx
                    )
                    
                    # 写入离子数信息
                    ion_count = 0 if -1 in o_dict else len(o_dict)
                    f_idx.write(f"{ion_count:>8}{md_step:>16}")
                    f_h.write(f"{ion_count:>8}{md_step:>16}")
                    f_o.write(f"{ion_count:>8}{md_step:>16}")
                    
                    # 写入离子数据
                    if ion_count > 0:
                        for o_id in o_dict:
                            if o_id == -1: 
                                continue
                            write_ion_data(
                                f_idx, f_h, f_o, 
                                o_id, o_coord_dict[o_id], 
                                o_dict[o_id], h_coord_dict[o_id],
                                scheme, dimensions  
                            )
                    
                    # 写入占位符
                    remaining = ion_num_upper - ion_count
                    for _ in range(remaining):
                        write_placeholder(f_idx, f_h, f_o, scheme)
                    
                    # 结束一行
                    f_idx.write("\n")
                    f_h.write("\n")
                    f_o.write("\n")
                    
                    # 更新进度条
                    if frame_count % 10 == 0:
                        pbar.update(10)
                
                except Exception as e:
                    print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        
        print("\n处理完成!")
        print(f"处理了 {frame_count} 帧")
        print(f"索引文件: {f_idx.name}")
        print(f"氢坐标文件: {f_h.name}")
        print(f"氧坐标文件: {f_o.name}")
    
    except Exception as e:
        print(f"处理轨迹时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保文件关闭
        if 'f_idx' in locals(): f_idx.close()
        if 'f_h' in locals(): f_h.close()
        if 'f_o' in locals(): f_o.close()