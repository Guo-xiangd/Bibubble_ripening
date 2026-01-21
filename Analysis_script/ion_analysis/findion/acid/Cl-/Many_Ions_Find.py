#!/usr/bin/env python
# coding: utf-8

import MDAnalysis as mda
import numpy as np
import os
import argparse

def trj_info(u):
    """使用MDAnalysis获取轨迹信息"""
    atom_nums = u.atoms.n_atoms
    frames = len(u.trajectory)
    return atom_nums, frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ion coordinates from trajectory using MDAnalysis.')
    
    parser.add_argument('trj_path', type=str, help='Path to the trajectory directory')
    parser.add_argument('save_path', type=str, help='Path to save output file')
    parser.add_argument('trj_file', type=str, help='Trajectory file name (e.g. centered_bubble1.lammpstrj)')
    parser.add_argument('findpart', type=int, help='Which part (block index) to process')
    parser.add_argument('choose_part', type=int, help='Total number of blocks')
    parser.add_argument('ion_num', type=int, help='Number of ions to extract')
    parser.add_argument('ion_type', type=str, help='Ion type: Na or Cl')
    parser.add_argument('time_section', type=str, help='the wanted time section like 0-8ns')

    args = parser.parse_args()
    
    trj_path = args.trj_path
    save_path = args.save_path
    trj_file = args.trj_file
    findpart = args.findpart
    choose_part = args.choose_part
    ion_num = args.ion_num
    ion_type = args.ion_type
    time_section = args.time_section
    
    # 离子类型与LAMMPS原子类型的映射
    ion_type_map = {'Na': 4, 'Cl': 5}
    atom_type = ion_type_map[ion_type]

    step = 1
    trj_name = trj_file.split('.')[0]

    # 输出文件命名
    out_filename = f"{ion_type}_dump{step}_{trj_name}_{findpart}_{time_section}.txt"
    ion_xyzfile = open(os.path.join(save_path, out_filename), 'w')
    ion_xyzfile.write('# %6s%16s' % ('Ions', 'MD_step'))

    for i in range(ion_num):
        ion_xyzfile.write('%8s' % (ion_type + str(i + 1)))
        ion_xyzfile.write('%16s%16s%16s' % ('X', 'Y', 'Z'))

    ion_xyzfile.write('\n')

    # 创建Universe对象
    u = mda.Universe(os.path.join(trj_path, trj_file), format='LAMMPSDUMP')
    
    # 获取轨迹信息
    atom_nums, total_frames = trj_info(u)
    
    # 计算要处理的帧范围
    frames_per_part = total_frames // choose_part
    start_frame = findpart * frames_per_part
    end_frame = min((findpart + 1) * frames_per_part, total_frames)

    print(f"Processing frames {start_frame} to {end_frame} (step={step})")

    # 选择目标离子
    ions = u.select_atoms(f'type {atom_type}')

    for ts in u.trajectory[start_frame:end_frame:step]:
        md_step = ts.frame * 1000
        
        ion_ids = ions.ids
        ion_coords = ions.positions

        ion_xyzfile.write('%8d%16d' % (len(ion_ids), md_step))
        for ion_id, coord in zip(ion_ids, ion_coords):
            ion_xyzfile.write('%8d' % ion_id)
            for val in coord:
                ion_xyzfile.write('%16.8f' % val)
        ion_xyzfile.write('\n')

    ion_xyzfile.close()
    print("Processing completed successfully.")