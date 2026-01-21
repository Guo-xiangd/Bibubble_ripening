#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2 migration analysis tool  (Python3.7+ compatible)
Usage example:
python n2_migration_analysis.py --trj file1:all file2:first:2000 file3:last:1000
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Dict, Tuple, Optional

# ---------- Input Parameters ----------

targeted_ID = 1

# ---------- Utility Functions ----------
def read_single_frame(f):
    line = f.readline()
    if not line:
        return None
    assert line.strip() == "ITEM: TIMESTEP"
    timestep = int(f.readline())
    assert f.readline().strip() == "ITEM: NUMBER OF ATOMS"
    natoms = int(f.readline())
    assert "ITEM: BOX BOUNDS" in f.readline()
    box = []
    for _ in range(3):
        box.append(tuple(map(float, f.readline().split())))
    header = f.readline().strip()
    assert header.startswith("ITEM: ATOMS")
    atoms = []
    for _ in range(natoms):
        parts = f.readline().split()
        atm_id = int(parts[0])
        cluster = int(parts[1])
        typ = int(parts[2])
        x, y, z = map(float, parts[3:6])
        atoms.append((atm_id, cluster, typ, x, y, z))
    return timestep, natoms, box, atoms


def iter_frames_by_policy(path: str, policy: str, count: Optional[int] = None):
    with open(path, 'r') as f:
        if policy == 'all':
            while True:
                frm = read_single_frame(f)
                if frm is None:
                    break
                yield frm
        else:
            frames = []
            while True:
                frm = read_single_frame(f)
                if frm is None:
                    break
                frames.append(frm)
            n = len(frames)
            if policy == 'first':
                sel = frames[:count]
            elif policy == 'last':
                sel = frames[-count:]
            else:
                raise ValueError(f"unknown policy {policy}")
            for frm in sel:
                yield frm


def parse_mol_id(atom_id: int) -> int:
    if atom_id % 2 == 0:
        return atom_id // 2
    else:
        return (atom_id + 1) // 2


def build_mol_to_cluster_map(atoms: List[tuple]) -> Dict[int, int]:
    mol_map = {}
    for atm_id, cluster, typ, x, y, z in atoms:
        mol_id = parse_mol_id(atm_id)
        if mol_id not in mol_map:
            mol_map[mol_id] = cluster
    return mol_map


# ---------- Rewrite analyze() function ----------
def analyze(trj_specs: List[Tuple[str, str, Optional[int]]],
            out_log: str = "N2_migration.log",
            min_valid: float = 0.5):
    """
    trj_specs: [(path, policy, count), ...]
    min_valid: Minimum interval to confirm a detachment/merge event (ns)
    """
    # 1. Get first frame, find molecules initially in bubble-2
    first_frame = None
    for path, pol, cnt in trj_specs:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        for frm in iter_frames_by_policy(path, pol, cnt):
            first_frame = frm
            break
        if first_frame is not None:
            break
    if first_frame is None:
        raise RuntimeError("No frame found!")
    ts0, natoms, box0, atoms0 = first_frame
    mol_map0 = build_mol_to_cluster_map(atoms0)
    target_mols = sorted([m for m, c in mol_map0.items() if c == targeted_ID])
    print(f"Initial bubble-{targeted_ID} molecules ({len(target_mols)}): {target_mols}")

    # 2. Initialize
    events = {m: {"detach2": [], "merge2": [], "detach1": [], "merge1": []} for m in target_mols}

    # Maintain "pending" and "last confirmed" time for each molecule
    # Structure: mol -> {"pend_d2": t or None, "last_m2": t or None, ...}
    state = {}
    for m in target_mols:
        state[m] = {"pend_d2": None, "last_m2": None,   # bubble-2 Exit-Return
                    "pend_d1": None, "last_m1": None}   # bubble-1 Enter-Exit

    # 3. Iterate through merged trajectory
    frame_idx = 0
    last_cluster_map = mol_map0
    for path, pol, cnt in trj_specs:
        for frm in iter_frames_by_policy(path, pol, cnt):
            ts, natoms, box, atoms = frm
            t_ns = frame_idx * 0.001
            curr_cluster_map = build_mol_to_cluster_map(atoms)

            for m in target_mols:
                prev_c = last_cluster_map[m]
                curr_c = curr_cluster_map[m]

                # -------- Bubble-2 Exit -> Return --------
                if prev_c == 2 and curr_c == 3:
                    # Leaving bubble-2: Record pending detachment
                    state[m]["pend_d2"] = t_ns

                elif prev_c == 3 and curr_c == 2:
                    # Returning to bubble-2: Check for pending detachment
                    if state[m]["pend_d2"] is not None:
                        delta = t_ns - state[m]["pend_d2"]
                        if delta >= min_valid:
                            # Confirm this pair
                            events[m]["detach2"].append(state[m]["pend_d2"])
                            events[m]["merge2"].append(t_ns)
                            state[m]["last_m2"] = t_ns
                        # Clear pending regardless of confirmation
                        state[m]["pend_d2"] = None

                # -------- Bubble-1 Entry -> Exit --------
                elif prev_c == 3 and curr_c == 1:
                    # Entering bubble-1: Record pending merge
                    state[m]["pend_m1"] = t_ns

                elif prev_c == 1 and curr_c == 3:
                    # Leaving bubble-1: Check for pending merge
                    if state[m].get("pend_m1") is not None:
                        delta = t_ns - state[m]["pend_m1"]
                        if delta >= min_valid:
                            events[m]["merge1"].append(state[m]["pend_m1"])
                            events[m]["detach1"].append(t_ns)
                            state[m]["last_d1"] = t_ns
                        state[m]["pend_m1"] = None

            last_cluster_map = curr_cluster_map
            frame_idx += 1

    # 4. Write log (column order identical to original script)
    header_order = []
    for i in range(1, 6):
        for stub in ("detach1", "merge1", "detach2", "merge2"):
            header_order.append(f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}_{stub}")

    with open(out_log, 'w') as fo:
        fo.write("#molID\t" + "\t".join(header_order) + "\n")
        for m in target_mols:
            parts = [str(m)]
            for stub in header_order:
                idx = int(stub[0])
                key = stub.split('_', 1)[1]  # detach1/merge1/detach2/merge2
                lst = events[m][key]
                val = lst[idx - 1] if idx - 1 < len(lst) else -1
                parts.append(f"{val:.3f}" if val != -1 else "-1")
            fo.write("\t".join(parts) + "\n")
            
    # ===== New: Output cluster ID evolution over time =====
    print("Recording cluster evolution...")
    n_frames = frame_idx
    # Pre-allocate memory: mol -> [cluster_0, cluster_1, ...]
    evo = {m: [0] * n_frames for m in target_mols}
    # Rescan trajectory (Minimize memory usage: only record target molecules)
    iframe = 0
    for path, pol, cnt in trj_specs:
        for frm in iter_frames_by_policy(path, pol, cnt):
            ts, natoms, box, atoms = frm
            curr_map = build_mol_to_cluster_map(atoms)
            for m in target_mols:
                evo[m][iframe] = curr_map[m]
            iframe += 1

    # Write file
    evo_file = f"cluster{targeted_ID}_evolution.log"
    with open(evo_file, 'w') as fe:
        # Header
        fe.write("#time(ns)\t" + "\t".join(map(str, target_mols)) + "\n")
        for i in range(n_frames):
            t_ms = i * 0.001
            fe.write(f"{t_ms:.3f}")
            for m in target_mols:
                fe.write(f"\t{evo[m][i]}")
            fe.write("\n")
    print(f"Cluster evolution saved to {evo_file}")
    
    print(f"Pair-validation (≥ {min_valid} ns) analysis done. Results saved to {out_log}")


# ---------- Command Line ----------
def main():
    parser = argparse.ArgumentParser(description="N2 migration analysis for LAMMPS trajectories")
    parser.add_argument("--trj", nargs="+", required=True,
                        help="Trajectory specs: path:policy[:count]  policy=all/first/last")
    parser.add_argument("-o", "--output", default="N2_migration.log", help="Output log file")
    args = parser.parse_args()
    specs = []
    for item in args.trj:
        parts = item.split(":")
        path = parts[0]
        pol = parts[1]
        cnt = None if len(parts) < 3 else int(parts[2])
        specs.append((path, pol, cnt))
    analyze(specs, args.output)


if __name__ == "__main__":
    main()