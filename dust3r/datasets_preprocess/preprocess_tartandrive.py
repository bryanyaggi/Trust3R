#!/usr/bin/env python3
"""Preprocess TartanDrive 2.0 runs (KITTI-like export) into the DUSt3R/MASt3R Waymo format.

Mirrors preprocess_rellis.py so the output loads with dust3r.datasets.waymo.Waymo unchanged. Output layout, one
dataset ROOT per split (all runs -> --split):

    <output_dir>/<split>/waymo_pairs.npz            scenes (= run names), frames, pairs (scene_id, im1, im2)
    <output_dir>/<split>/<run>/<frame>.jpg          RGB, rectified left MultiSense camera
    <output_dir>/<split>/<run>/<frame>.exr          float32 depth (camera z, 0 = no LiDAR)
    <output_dir>/<split>/<run>/<frame>.npz          intrinsics, cam2world (OpenCV), distortion (zeros: rectified)
    <output_dir>/<split>/<run>/hood_mask.png        pixels zeroed in every depth map (ATV hood), output resolution

Per frame:
  * one scan per image (no merging): velodyne_0 by default (the level unit, whose frame is the vehicle frame; the
    camera extrinsic was checked with it); --lidars velodyne_0,velodyne_1 adds the pitched unit (checked by eye only);
  * every point is deskewed to the image time with SuperOdometry (per-point time from azimuth), then projected
    with the released extrinsic + --roll_deg (default +0.875 deg, from a stereo-vs-LiDAR sweep);
  * z-buffered (nearest point per pixel), points closer than --min_range dropped, hood pixels zeroed;
  * cam2world = T_world<-vehicle(image time) @ T_vehicle<-cam, world origin moved to the run's first frame;
  * frames the extractor filled by copying a neighbour (image or any used LiDAR) are skipped.

Pairs: TartanDrive speed varies (0-15 m/s), so pairs are chosen by camera baseline, not frame gap: for each
frame and each target in --baselines, the first later frame whose camera centre is at least that far away,
within --max_gap frames. Duplicates (fast driving: several targets hit the same frame) are dropped. The default,
0.75 m, gives one pair per view-1 frame (KITTI-style); 4,343 pairs for gupta_2023-09-14-11-12-19.

TartanDrive 2.0 has no official split; all runs go to --split (default test).

Check: with the defaults, gupta_2023-09-14-11-12-19 alone gives these files, byte for byte, as the data evaluated
in the RELLIS-3D fine-tuning project. Compare with --fingerprint, which should print:
    pairs 4343  frames 4345  pairs sha256 9aee0bc268de801d  LiDAR pixels 50476680  mean depth 19.1270 m

Conventions (checked 2026-11-20 on gupta_2023-09-14-11-12-19 and turnpike_afternoon_fall):
  * calibration/extrinsics.yaml entries are poses: X_from = T * X_to. vehicle = velodyne_0.tar frame
    (bag /velodyne_1); velodyne_1.tar is bag /velodyne_2 (pitched 10 deg down), extrinsic entry velodyne_2.
  * The released camera extrinsic needs +0.875 deg roll about the optical axis (stereo vs LiDAR sweep);
    pitch and yaw are right as released.
  * Frames sit on a 10 Hz grid starting at the first image stamp; per-frame stamps are not saved.
    super_odom is 10 rows per frame, 13 columns (xyz, quat xyzw, linear vel, angular vel) = T_odom<-vehicle,
    with duplicated stamps at rows the extractor filled.
  * Velodyne points are stored in firing order: clockwise from the rear (azimuth +180 deg) to -180 deg.
  * Timing: scan start = grid time + LIDAR_OFFSET (puts the forward sweep on the grid time); the image is at
    grid time + CAM_OFFSET. Stereo vs LiDAR (single frame, the GT evaluation uses) peaks at 0 ms.

Input (https://github.com/castacks/tartan_drive_2.0, MIT license): per run, from the public bucket's
kitti/all_topics/<run>/, these files, keeping their paths under one folder per run (tars are read in place, not
extracted). With velodyne_0 only, about 7.1 GB for gupta_2023-09-14-11-12-19:
    image_left_color.tar  velodyne_0.tar  super_odom/odometry.npy  super_odom/timestamps.txt  data_extraction.log
    (velodyne_1.tar too for --lidars velodyne_0,velodyne_1)

    B=https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2
    R=gupta_2023-09-14-11-12-19
    for f in image_left_color.tar velodyne_0.tar super_odom/odometry.npy super_odom/timestamps.txt \\
             data_extraction.log; do
      curl -f --create-dirs -C - -o "TartanDrive2/$R/$f" "$B/kitti/all_topics/$R/$f"
    done

Usage, from the repository root:
    python dust3r/datasets_preprocess/preprocess_tartandrive.py --tartandrive_dir TartanDrive2 \\
        --runs gupta_2023-09-14-11-12-19 --output_dir data/tartandrive_waymo_512 --workers 8
    python dust3r/datasets_preprocess/preprocess_tartandrive.py --tartandrive_dir TartanDrive2 \\
        --runs gupta_2023-09-14-11-12-19 --output_dir data/tartandrive_waymo_512 --pairs_only --baselines 0.5,1.0
    python dust3r/datasets_preprocess/preprocess_tartandrive.py --output_dir data/tartandrive_waymo_512 --fingerprint
"""
import argparse
import io
import os
import os.path as osp
import re
import tarfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
from scipy.ndimage import median_filter, minimum_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402
from tqdm import tqdm  # noqa: E402

import path_to_root  # noqa: E402,F401
from dust3r.datasets.utils import cropping  # noqa: E402


# ---------------------------------------------------------------- TartanDrive 2.0 I/O and calibration
# rectified left colour camera (multisense/left/image_rect_color camera_info P)
K = np.array([[477.6049499511719, 0, 499.5], [0, 477.6049499511719, 252.0], [0, 0, 1]])
W_NATIVE, H_NATIVE = 1024, 544
SCAN_PERIOD = 0.1
LIDAR_OFFSET = -0.05
CAM_OFFSET = 0.0
ROLL_FIX_DEG = 0.875


def pose(t, q):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(q).as_matrix()
    M[:3, 3] = t
    return M


T_VEH_HEAD = pose([-0.0949, 0.0641, -0.136], [-0.0119, 0.1131, 0.0003, 0.9935])
T_HEAD_CAM = pose([0.005, 0.105, 0.0], [-0.5, 0.5, -0.5, 0.5])
T_VEH_CAM = T_VEH_HEAD @ T_HEAD_CAM  # as released
T_VEH_V2 = pose([-0.115, 0.0, 0.205], [0.0, 0.0871558, 0.0, 0.9961947])
LIDAR_TOPIC = {"velodyne_0": "/velodyne_1/velodyne_points", "velodyne_1": "/velodyne_2/velodyne_points"}
IMAGE_TOPIC = "/multisense/left/image_rect_color"


def perturbed_extrinsic(R_err):
    """Released T_veh<-cam with an extra rotation about the camera's optical axes (x right = pitch,
    y down = yaw, z forward = roll)."""
    T = T_VEH_CAM.copy()
    T[:3, :3] = T[:3, :3] @ R_err
    return T


def camera_extrinsic(roll_deg=ROLL_FIX_DEG):
    """T_veh<-cam with the roll correction."""
    return perturbed_extrinsic(Rotation.from_euler("z", roll_deg, degrees=True).as_matrix())


class TarFrames:
    def __init__(self, path):
        self.tar = tarfile.open(path)
        self.mem = {m.name.split("/")[-1]: m for m in self.tar.getmembers() if m.isfile()}

    def raw(self, name):
        return self.tar.extractfile(self.mem[name]).read()

    def points(self, i):
        return np.load(io.BytesIO(self.raw(f"{i:06d}.npy"))).astype(np.float64)

    def image(self, i):
        return cv2.imdecode(np.frombuffer(self.raw(f"{i:06d}.png"), np.uint8), cv2.IMREAD_COLOR)


class Odometry:
    def __init__(self, run):
        o = np.load(run / "super_odom/odometry.npy")
        t = np.loadtxt(run / "super_odom/timestamps.txt")
        # rows are message stamps matched to a 100 Hz grid; filled rows repeat a stamp -> keep strictly increasing
        n_frames = len(t) // 10
        keep = np.r_[True, np.diff(t) > 0]
        keep &= t >= np.maximum.accumulate(t)
        self.t, self.p = t[keep], o[keep, :3]
        self.slerp = Slerp(self.t, Rotation.from_quat(o[keep, 3:7]))
        # frame i is on the 10 Hz grid that starts at the first image stamp
        self.t_frame = t[0] + 0.1 * np.arange(n_frames)

    def T(self, t):
        """T_world<-vehicle at times t (array)."""
        t = np.clip(np.atleast_1d(t), self.t[0], self.t[-1])
        M = np.tile(np.eye(4), (len(t), 1, 1))
        M[:, :3, :3] = self.slerp(t).as_matrix()
        M[:, :3, 3] = np.stack([np.interp(t, self.t, self.p[:, k]) for k in range(3)], 1)
        return M


def filled_frames(run):
    """Frames the extractor filled by copying a neighbour, per topic."""
    out = {}
    for line in open(run / "data_extraction.log"):
        m = re.match(r"\s*-- (\S+) filled missing frame (\d+) <-", line)
        if m:
            out.setdefault(m.group(1), set()).add(int(m.group(2)))
    return out


def point_times(p):
    """Seconds after scan start, from azimuth (clockwise sweep starting at the rear)."""
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return ((180.0 - az) % 360.0) / 360.0 * SCAN_PERIOD


def to_world(odo, p, t_frame, lidar_offset=LIDAR_OFFSET, deskew=True):
    """Vehicle-frame scan -> world. Scan start = t_frame + lidar_offset."""
    if deskew:
        Ts = odo.T(t_frame + lidar_offset + point_times(p))
        return np.einsum("nij,nj->ni", Ts[:, :3, :3], p) + Ts[:, :3, 3]
    T = odo.T(t_frame + lidar_offset + SCAN_PERIOD / 2)[0]
    return p @ T[:3, :3].T + T[:3, 3]


def hood_mask(imgs, n_frames, n_samples=150, margin=4):
    """Pixels covered by the ATV hood (True), at native resolution.

    The hood's brightness changes with the sun, so it is found as the strongest edges near the bottom of the
    mean image over n_samples frames (scene edges average out): per column, the top hood edge in rows >= 440,
    smoothed, then extended down to the bottom and up by `margin` px.
    """
    ids = np.linspace(0, n_frames - 1, n_samples).astype(int)
    mean = np.mean([cv2.cvtColor(imgs.image(i), cv2.COLOR_BGR2GRAY).astype(np.float32) for i in ids], 0)
    g = cv2.GaussianBlur(mean, (3, 3), 0)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    thr = 1.5 * np.percentile(grad[300:460], 99.9)
    top = np.full(W_NATIVE, H_NATIVE)
    for x in range(W_NATIVE):
        rows = np.flatnonzero(grad[440:, x] > thr)
        if len(rows):
            top[x] = 440 + rows[0]
    top = median_filter(minimum_filter1d(top, 9), 15)
    mask = np.zeros((H_NATIVE, W_NATIVE), bool)
    for x in np.flatnonzero(top < H_NATIVE):
        mask[max(top[x] - margin, 0):, x] = True
    return mask


# ---------------------------------------------------------------- converter
def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tartandrive_dir",
                   help="directory holding one folder per run (the KITTI-like export, tars not extracted)")
    p.add_argument("--runs", default=None, help="comma list of run folders (default: all in --tartandrive_dir)")
    p.add_argument("--output_dir", required=True, help="one Waymo-format ROOT per split is written below it")
    p.add_argument("--split", default="test")
    p.add_argument("--resolution", type=int, default=512, help="long-side size in px (default 512, as in Waymo)")
    p.add_argument("--lidars", default="velodyne_0", help="comma list of velodyne_0, velodyne_1")
    p.add_argument("--roll_deg", type=float, default=ROLL_FIX_DEG,
                   help="extra camera roll about the optical axis on top of the released extrinsic (deg)")
    p.add_argument("--min_range", type=float, default=1.0, help="drop LiDAR points closer than this (m)")
    p.add_argument("--no_deskew", action="store_true", help="use one pose per scan instead of per point")
    p.add_argument("--baselines", default="0.75",
                   help="comma list of target camera baselines (m); 0.75 = one pair per frame, as evaluated. "
                        "RELLIS gaps 2-10 at ~13 cm/frame span 0.26-1.3 m")
    p.add_argument("--max_gap", type=int, default=30, help="max frames between a pair's images (3 s)")
    p.add_argument("--max_frames", type=int, default=None, help="debug: only the first N frames of each run")
    p.add_argument("--pairs_only", action="store_true", help="only rebuild waymo_pairs.npz from existing frames")
    p.add_argument("--fingerprint", action="store_true",
                   help="only print a content fingerprint of <output_dir>/<split> (compare with the docstring)")
    p.add_argument("--workers", type=int, default=8)
    return p


# ---------------------------------------------------------------- per-run setup
def run_frames(run, lidars, max_frames):
    """Frame indices to convert: on the grid, not filled by the extractor in any used stream."""
    odo = Odometry(run)
    filled = filled_frames(run)
    skip = set(filled.get(IMAGE_TOPIC, set()))
    for lidar in lidars:
        skip |= filled.get(LIDAR_TOPIC[lidar], set())
    idxs = [i for i in range(len(odo.t_frame)) if i not in skip][:max_frames]
    return idxs, len(skip)


def write_hood_mask(run, out_dir, out_hw):
    path = osp.join(out_dir, "hood_mask.png")
    if not osp.isfile(path):
        imgs = TarFrames(run / "image_left_color.tar")
        n = len([m for m in imgs.mem if m.endswith(".png")])
        mask = hood_mask(imgs, n)
        mask = cv2.resize(mask.astype(np.uint8), out_hw[::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, mask * 255)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0


# ---------------------------------------------------------------- per-frame conversion (worker side)
_W = {}  # per-worker cache: run -> (odometry, image tar, lidar tars, hood mask, world origin)


def _worker_state(run, out_dir, lidars, resolution):
    if run not in _W:
        odo = Odometry(run)
        origin = odo.T(odo.t_frame[:1] + CAM_OFFSET)[0][:3, 3]
        hw = output_hw(resolution)
        _W[run] = (odo, TarFrames(run / "image_left_color.tar"),
                   [TarFrames(run / f"{lidar}.tar") for lidar in lidars],
                   cv2.imread(osp.join(out_dir, "hood_mask.png"), cv2.IMREAD_GRAYSCALE) > 0, origin)
        assert _W[run][3].shape == hw, (_W[run][3].shape, hw)
    return _W[run]


def output_hw(resolution):
    image = PIL.Image.new("RGB", (W_NATIVE, H_NATIVE))
    out_res = (resolution, 1) if W_NATIVE > H_NATIVE else (1, resolution)
    image, _, _ = cropping.rescale_image_depthmap(image, None, K, out_res)
    return image.size[1], image.size[0]


def process_frame(job, lidars, resolution, min_range, roll_deg, deskew):
    run, idx, out_dir = job
    name = f"{idx:06d}"
    if osp.isfile(osp.join(out_dir, name + ".npz")):  # npz is written last
        return
    odo, imgs, velos, hood, origin = _worker_state(run, out_dir, lidars, resolution)

    # image (+ intrinsics) at the output resolution
    image = PIL.Image.fromarray(cv2.cvtColor(imgs.image(idx), cv2.COLOR_BGR2RGB))
    out_res = (resolution, 1) if W_NATIVE > H_NATIVE else (1, resolution)
    image, _, K_out = cropping.rescale_image_depthmap(image, None, K, out_res)
    image.save(osp.join(out_dir, name + ".jpg"), quality=80)
    W, H = image.size

    # scan(s) -> vehicle frame -> world (deskewed) -> camera at the image time
    scans = []
    for lidar, velo in zip(lidars, velos):
        p = velo.points(idx)
        p = p[np.linalg.norm(p, axis=1) >= min_range]
        if lidar == "velodyne_1":
            p = p @ T_VEH_V2[:3, :3].T + T_VEH_V2[:3, 3]
        scans.append(p)
    pts = np.concatenate(scans)
    t_frame = odo.t_frame[idx]
    Xw = to_world(odo, pts, t_frame, LIDAR_OFFSET, deskew)
    world_from_cam = odo.T(t_frame + CAM_OFFSET)[0] @ camera_extrinsic(roll_deg)
    cam_from_world = np.linalg.inv(world_from_cam)
    pc = Xw @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
    pc = pc[pc[:, 2] > 0]
    uv = pc @ K_out.T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)  # OpenCV convention: pixel centres at integers
    v = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], pc[inside, 2]

    # z-buffer: nearest point per pixel; hood pixels see the hood, not the LiDAR's ground behind it
    order = np.argsort(z)
    pix = v[order] * W + u[order]
    _, first = np.unique(pix, return_index=True)
    sel = order[first]
    depthmap = np.zeros((H, W), np.float32)
    depthmap[v[sel], u[sel]] = z[sel]
    depthmap[hood] = 0
    cv2.imwrite(osp.join(out_dir, name + ".exr"), depthmap)

    world_from_cam[:3, 3] -= origin
    np.savez(osp.join(out_dir, name + ".npz"), intrinsics=K_out, cam2world=world_from_cam, distortion=np.zeros(5))


def extract_frames(args, runs, lidars):
    jobs = []
    hw = output_hw(args.resolution)
    for run in runs:
        out_dir = osp.join(args.output_dir, args.split, run.name)
        os.makedirs(out_dir, exist_ok=True)
        hood = write_hood_mask(run, out_dir, hw)
        idxs, n_skip = run_frames(run, lidars, args.max_frames)
        print(f"{run.name}: {len(idxs)} frames ({n_skip} filled by the extractor skipped), "
              f"hood mask {hood.mean() * 100:.1f}% of pixels")
        jobs += [(run, idx, out_dir) for idx in idxs]
    fn = partial(process_frame, lidars=lidars, resolution=args.resolution, min_range=args.min_range,
                 roll_deg=args.roll_deg, deskew=not args.no_deskew)
    # jobs are ordered by run and frame, so each worker mostly reuses one run's open tars
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap(fn, jobs, chunksize=16), total=len(jobs), desc="frames"):
            pass


# ---------------------------------------------------------------- pairs
def make_pairs(args, runs):
    targets = sorted(float(b) for b in args.baselines.split(","))
    root = osp.join(args.output_dir, args.split)
    scenes, frames, pairs = [], [], []
    for run in runs:
        run_dir = osp.join(root, run.name)
        idxs = sorted(int(f[:-4]) for f in os.listdir(run_dir) if f.endswith(".npz"))
        if not idxs:
            continue
        scene_id = len(scenes)
        scenes.append(run.name)
        frame_id = {}
        for idx in idxs:
            frame_id[idx] = len(frames)
            frames.append(f"{idx:06d}")
        centres = np.array([np.load(osp.join(run_dir, f"{i:06d}.npz"))["cam2world"][:3, 3] for i in idxs])
        idx_arr = np.array(idxs)
        n_run = 0
        for a, i in enumerate(idxs):
            later = np.flatnonzero((idx_arr > i) & (idx_arr <= i + args.max_gap))
            if not len(later):
                continue
            dist = np.linalg.norm(centres[later] - centres[a], axis=1)
            partners = set()
            for b in targets:
                hit = np.flatnonzero(dist >= b)
                if len(hit):
                    partners.add(idxs[later[hit[0]]])
            for j in sorted(partners):
                pairs.append((scene_id, frame_id[i], frame_id[j]))
                n_run += 1
        print(f"{run.name}: {n_run} pairs over {len(idxs)} frames")
    np.savez(osp.join(root, "waymo_pairs.npz"), scenes=np.array(scenes), frames=np.array(frames),
             pairs=np.array(pairs, dtype=np.int64))
    print(f"{args.split}: {len(pairs)} pairs over {len(frames)} frames in {len(scenes)} scenes "
          f"(baselines {targets} m, max gap {args.max_gap} frames)")


def fingerprint(args):
    """Content fingerprint: robust to npz zip timestamps and EXR encoder versions, unlike file hashes."""
    import hashlib
    root = osp.join(args.output_dir, args.split)
    p = np.load(osp.join(root, "waymo_pairs.npz"))
    h = hashlib.sha256(p["pairs"].astype(np.int64).tobytes() + "".join(p["frames"]).encode()
                       + "".join(p["scenes"]).encode())
    n_pix, z_sum = 0, 0.0
    for scene in p["scenes"]:
        names = sorted(f[:-4] for f in os.listdir(osp.join(root, scene)) if f.endswith(".exr"))
        for name in tqdm(names, desc=f"{scene} depth"):
            d = cv2.imread(osp.join(root, scene, name + ".exr"), cv2.IMREAD_UNCHANGED)
            n_pix += int((d > 0).sum())
            z_sum += float(d[d > 0].astype(np.float64).sum())
    print(f"pairs {len(p['pairs'])}  frames {len(p['frames'])}  pairs sha256 {h.hexdigest()[:16]}  "
          f"LiDAR pixels {n_pix}  mean depth {z_sum / max(n_pix, 1):.4f} m")


def run(args):
    if args.fingerprint:
        return fingerprint(args)
    if args.tartandrive_dir is None:
        raise SystemExit("--tartandrive_dir is required")
    root = Path(args.tartandrive_dir)
    names = args.runs.split(",") if args.runs else sorted(d.name for d in root.iterdir() if d.is_dir())
    runs = [root / n for n in names]
    lidars = args.lidars.split(",")
    assert all(lidar in LIDAR_TOPIC for lidar in lidars), lidars
    if not args.pairs_only:
        extract_frames(args, runs, lidars)
    make_pairs(args, runs)
    print("Done! data at", args.output_dir)


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
