#!/usr/bin/env python3
"""Preprocess RELLIS-3D into the DUSt3R/MASt3R Waymo format.

Mirrors DUSt3R's datasets_preprocess/preprocess_waymo.py (https://github.com/naver/dust3r) so the output loads
with dust3r.datasets.waymo.Waymo unchanged: one Ouster scan per image (no merging), projected
to a sparse depth map. Output layout, one dataset ROOT per LiDAR split:

    <output_dir>/<split>/waymo_pairs.npz            scenes, frames, pairs (scene_id, im1, im2)
    <output_dir>/<split>/<seq>/<frame>.jpg          RGB
    <output_dir>/<split>/<seq>/<frame>.exr          float32 depth (camera z, 0 = no LiDAR)
    <output_dir>/<split>/<seq>/<frame>.npz          intrinsics, cam2world (OpenCV), distortion
    <output_dir>/<split>/<seq>/<frame>_label.png    uint8 LiDAR semantic id at depth pixels (extra)

Differences from the Waymo script:
  * --resolution: long side in px, default 512 like Waymo (512x320); 0 keeps the native
    1920x1200 images. LiDAR is projected directly at the output resolution. The dust3r loader
    resizes depth with nearest neighbour, which preserves the stored fill fraction, so native
    output gives only ~0.3% depth coverage at 224 (512: 4.5%, 384: 8%).
  * z-buffered: the nearest point wins per pixel (Waymo lets whichever point comes last win).
  * no-return points (range 0) and points closer than --min_range are dropped.
  * splits follow the LiDAR pt_{train,val,test}.lst (contiguous blocks, no pair leakage).

Conventions (verified 2026-11-10/11):
  * transforms.yaml `os1_cloud_node-pylon_camera_node` is T_lidar<-cam.
  * poses.txt rows are T_world<-lidar (scan alignment: 0.1 m median NN vs 1.5 m for inverse).
  * scan i <-> image i; camera_info.txt is fx fy cx cy, no distortion.

Input: unzip these RELLIS-3D downloads (https://github.com/unmannedlab/RELLIS-3D) into one directory,
--rellis_dir, which then holds Rellis-3D/<seq>/..., Rellis_3D/<seq>/transforms.yaml and pt_{train,val,test}.lst:
    Rellis_3D_pylon_camera_node.zip                                 images
    Rellis_3D_os1_cloud_node_kitti_bin.zip                          Ouster scans (SemanticKITTI format)
    Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip    Ouster labels
    Rellis_3D_lidar_poses_20210614.zip                              poses.txt
    Rellis_3D_cam_intrinsic.zip                                     camera_info.txt
    Rellis_3D_cam2lidar_20210224.zip                                transforms.yaml (per sequence)
    Rellis_3D_lidar_split.zip                                       pt_{train,val,test}.lst

Usage, from the repository root:
    python dust3r/datasets_preprocess/preprocess_rellis.py --rellis_dir RELLIS --output_dir data/rellis_waymo_512
    python dust3r/datasets_preprocess/preprocess_rellis.py --rellis_dir RELLIS --output_dir OUT --pairs_only \\
        --pair_gaps 1,2,3
"""
import argparse
import os
import os.path as osp
import re
import shutil
from functools import partial
from multiprocessing import Pool

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from tqdm import tqdm  # noqa: E402

import path_to_root  # noqa: E402,F401
from dust3r.datasets.utils import cropping  # noqa: E402

SPLITS = ("train", "val", "test")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rellis_dir", required=True, help="directory the RELLIS-3D zips were extracted into")
    p.add_argument("--output_dir", required=True, help="one Waymo-format ROOT per split is written below it")
    p.add_argument("--resolution", type=int, default=512,
                   help="long-side size in px (default 512, as in Waymo); 0 keeps native 1920x1200")
    p.add_argument("--seqs", default="00000,00001,00002,00003,00004")
    p.add_argument("--min_range", type=float, default=1.0, help="drop LiDAR points closer than this (m)")
    p.add_argument("--pair_gaps", default="2,4,6,8,10",
                   help="frame gaps for pairs (10 Hz, ~13 cm/frame); each frame anchors one pair per gap")
    p.add_argument("--min_baseline", type=float, default=0.0,
                   help="drop pairs whose camera centres are closer than this (m); ~13%% of frames are stationary")
    p.add_argument("--max_frames", type=int, default=None, help="debug: only the first N frames of each seq")
    p.add_argument("--pairs_only", action="store_true", help="only rebuild waymo_pairs.npz from existing frames")
    p.add_argument("--workers", type=int, default=8)
    return p


# ---------------------------------------------------------------- RELLIS metadata
def load_seq_meta(rellis_dir, seq):
    d = osp.join(rellis_dir, "Rellis-3D", seq)
    fx, fy, cx, cy = np.loadtxt(osp.join(d, "camera_info.txt"))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    tf = yaml.safe_load(open(osp.join(rellis_dir, "Rellis_3D", seq, "transforms.yaml")))
    tf = tf["os1_cloud_node-pylon_camera_node"]
    lidar_from_cam = np.eye(4)
    q, t = tf["q"], tf["t"]
    lidar_from_cam[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    lidar_from_cam[:3, 3] = [t["x"], t["y"], t["z"]]
    poses = np.loadtxt(osp.join(d, "poses.txt")).reshape(-1, 3, 4)
    world_from_lidar = np.tile(np.eye(4), (len(poses), 1, 1))
    world_from_lidar[:, :3] = poses
    images = {}
    for f in os.listdir(osp.join(d, "pylon_camera_node")):
        m = re.match(r"frame(\d{6})-.*\.jpg$", f)
        if m:
            images[int(m.group(1))] = f
    assert len(images) == len(poses), f"{seq}: {len(images)} images vs {len(poses)} poses"
    return dict(dir=d, K=K, lidar_from_cam=lidar_from_cam, world_from_lidar=world_from_lidar, images=images)


def load_split_map(rellis_dir):
    """(seq, frame_idx) -> split, from the LiDAR split lists."""
    split_of = {}
    for split in SPLITS:
        for line in open(osp.join(rellis_dir, f"pt_{split}.lst")):
            if line.strip():
                seq, _, fname = line.split()[0].split("/")
                split_of[(seq, int(fname[:6]))] = split
    return split_of


def frame_name(idx):
    return f"{idx:06d}"


# ---------------------------------------------------------------- per-frame conversion
def process_frame(job, resolution, min_range):
    meta, idx, out_dir = job
    name = frame_name(idx)
    if osp.isfile(osp.join(out_dir, name + ".npz")):  # npz is written last
        return
    src_img = osp.join(meta["dir"], "pylon_camera_node", meta["images"][idx])
    K = meta["K"]

    # image (+ intrinsics) at the output resolution
    if not resolution:
        shutil.copyfile(src_img, osp.join(out_dir, name + ".jpg"))
        W, H = PIL.Image.open(src_img).size
        K_out = K.copy()
    else:
        image = PIL.Image.open(src_img).convert("RGB")
        W, H = image.size
        out_res = (resolution, 1) if W > H else (1, resolution)
        image, _, K_out = cropping.rescale_image_depthmap(image, None, K, out_res)
        image.save(osp.join(out_dir, name + ".jpg"), quality=80)
        W, H = image.size

    # LiDAR -> camera, drop no-returns and near points
    scan = np.fromfile(osp.join(meta["dir"], "os1_cloud_node_kitti_bin", f"{idx:06d}.bin"), np.float32)
    pts = scan.reshape(-1, 4)[:, :3].astype(np.float64)
    labels = np.fromfile(osp.join(meta["dir"], "os1_cloud_node_semantickitti_label_id", f"{idx:06d}.label"),
                         np.uint32) & 0xFFFF
    keep = np.linalg.norm(pts, axis=1) >= min_range
    pts, labels = pts[keep], labels[keep]
    cam_from_lidar = np.linalg.inv(meta["lidar_from_cam"])
    pc = pts @ cam_from_lidar[:3, :3].T + cam_from_lidar[:3, 3]
    front = pc[:, 2] > 0
    pc, labels = pc[front], labels[front]
    uv = pc @ K_out.T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)  # OpenCV convention: pixel centres at integers
    v = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, labels = u[inside], v[inside], pc[inside, 2], labels[inside]

    # z-buffer: nearest point per pixel
    order = np.argsort(z)
    pix = v[order] * W + u[order]
    _, first = np.unique(pix, return_index=True)
    sel = order[first]
    depthmap = np.zeros((H, W), np.float32)
    depthmap[v[sel], u[sel]] = z[sel]
    labelmap = np.zeros((H, W), np.uint8)
    labelmap[v[sel], u[sel]] = labels[sel]
    cv2.imwrite(osp.join(out_dir, name + ".exr"), depthmap)
    PIL.Image.fromarray(labelmap).save(osp.join(out_dir, name + "_label.png"))

    cam2world = meta["world_from_lidar"][idx] @ meta["lidar_from_cam"]
    np.savez(osp.join(out_dir, name + ".npz"), intrinsics=K_out, cam2world=cam2world, distortion=np.zeros(5))


def extract_frames(args, metas, split_of):
    jobs = []
    for seq, meta in metas.items():
        idxs = sorted(meta["images"])[:args.max_frames]
        for split in SPLITS:
            os.makedirs(osp.join(args.output_dir, split, seq), exist_ok=True)
        for idx in idxs:
            jobs.append((meta, idx, osp.join(args.output_dir, split_of[(seq, idx)], seq)))
    fn = partial(process_frame, resolution=args.resolution, min_range=args.min_range)
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(fn, jobs, chunksize=8), total=len(jobs), desc="frames"):
            pass


# ---------------------------------------------------------------- pairs
def make_pairs(args, metas, split_of):
    gaps = [int(g) for g in args.pair_gaps.split(",")]
    for split in SPLITS:
        root = osp.join(args.output_dir, split)
        scenes, frames, pairs = [], [], []
        n_dropped = 0
        for seq, meta in metas.items():
            seq_dir = osp.join(root, seq)
            idxs = sorted(int(f[:-4]) for f in os.listdir(seq_dir) if f.endswith(".npz"))
            if not idxs:
                continue
            scene_id = len(scenes)
            scenes.append(seq)
            frame_id = {}
            for idx in idxs:
                frame_id[idx] = len(frames)
                frames.append(frame_name(idx))
            # pairs must stay inside one contiguous block of this split
            present = set(idxs)
            centres = {i: (meta["world_from_lidar"][i] @ meta["lidar_from_cam"])[:3, 3] for i in idxs}
            for i in idxs:
                for g in gaps:
                    j = i + g
                    if not all(k in present for k in range(i + 1, j + 1)):
                        continue
                    if np.linalg.norm(centres[j] - centres[i]) < args.min_baseline:
                        n_dropped += 1
                        continue
                    pairs.append((scene_id, frame_id[i], frame_id[j]))
        if not pairs:
            continue
        np.savez(osp.join(root, "waymo_pairs.npz"), scenes=np.array(scenes), frames=np.array(frames),
                 pairs=np.array(pairs, dtype=np.int64))
        print(f"{split}: {len(pairs)} pairs over {len(frames)} frames in {len(scenes)} scenes"
              f" (gaps {gaps}, {n_dropped} dropped by min_baseline={args.min_baseline})")


def run(args):
    metas = {seq: load_seq_meta(args.rellis_dir, seq) for seq in args.seqs.split(",")}
    split_of = load_split_map(args.rellis_dir)
    if not args.pairs_only:
        extract_frames(args, metas, split_of)
    make_pairs(args, metas, split_of)
    print("Done! data at", args.output_dir)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
