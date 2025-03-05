from itertools import count
from multiprocessing import Process, Queue
from pathlib import Path
import time
import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    """
    Displays an image using OpenCV.
    
    Args:
        image (torch.Tensor): The image tensor to display.
        t (int): The time in milliseconds to wait for a key event. Default is 0.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def read_calib_file(filepath):
    """
    Reads a calibration file and parses it into a dictionary.
    
    Args:
        filepath (str): The path to the calibration file.
    
    Returns:
        dict: A dictionary containing the calibration data.
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def kitti_image_stream(queue, kittidir, sequence, stride, start_frame=0, end_frame=None, skip=0, max_images=200):
    """
    Streams images from the KITTI dataset into a queue for processing.
    
    Args:
        queue (Queue): The queue to store the images.
        kittidir (Path): The base directory of the KITTI dataset.
        sequence (str): The sequence number to process.
        stride (int): The stride to use when selecting frames.
        start_frame (int): The starting frame index. Default is 0.
        end_frame (int): The ending frame index. Default is None.
        skip (int): The number of frames to skip initially. Default is 0.
        max_images (int): The maximum number of images to process. Default is 200.
    """
    print("🚀 Starting kitti_image_stream...")  

    images_dir = kittidir / "dataset" / "sequences" / sequence
    all_images = sorted((images_dir / "image_2").glob("*.png"))

    if end_frame is None or end_frame > len(all_images):
        end_frame = len(all_images)

    image_list = all_images[start_frame:end_frame+1][skip::stride][:max_images]
    print(f"🔄 Loaded {len(image_list)} images from KITTI sequence {sequence}")  

    calib = read_calib_file(images_dir / "calib.txt")
    intrinsics = calib['P0'][[0, 5, 2, 6]]

    scale_factor = 1  

    for t, imfile in enumerate(image_list, start=start_frame):
        image_left = cv2.imread(str(imfile))
        if image_left is None:
            print(f"⚠️ Warning: Could not read {imfile}")
            continue

        H, W, _ = image_left.shape
        target_H = int(H * scale_factor) - int(H * scale_factor) % 16  # Ensure dimensions are divisible by 16
        target_W = int(W * scale_factor) - int(W * scale_factor) % 16

        image_left = cv2.resize(image_left, (target_W, target_H))

        queue.put((t, image_left, intrinsics))
        print(f"📤 Sent frame {t} to queue")  

    queue.put((-1, None, None))  
    print("✅ Finished sending images")

@torch.no_grad()
def run(cfg, network, kittidir, sequence, stride=1, start_frame=0, end_frame=None, viz=False, show_img=False, max_images=200):
    """
    Runs the SLAM algorithm on the KITTI dataset.
    
    Args:
        cfg: Configuration object.
        network (str): Path to the network model.
        kittidir (Path): The base directory of the KITTI dataset.
        sequence (str): The sequence number to process.
        stride (int): The stride to use when selecting frames. Default is 1.
        start_frame (int): The starting frame index. Default is 0.
        end_frame (int): The ending frame index. Default is None.
        viz (bool): Whether to visualize the process. Default is False.
        show_img (bool): Whether to show images during processing. Default is False.
        max_images (int): The maximum number of images to process. Default is 200.
    
    Returns:
        tuple: Estimated trajectory, timestamps, and group times.
    """
    torch.backends.cudnn.benchmark = True
    torch.cuda.amp.autocast(enabled=True)

    slam = None
    queue = Queue(maxsize=10)  # Same buffer_size
    frame_times = []
    
    reader = Process(target=kitti_image_stream, args=(queue, kittidir, sequence, stride, start_frame, end_frame, 0, max_images))
    reader.start()
    for step in count(start=1):
        start_time = time.time()

        # Ensure there are images available before `get()`
        while queue.qsize() == 0:
            print("⏳ Waiting for images in queue...")  # Confirm that `queue` is empty
            time.sleep(0.1)

        (t, image, intrinsics) = queue.get()
        print(f"📥 Received frame {t} from queue")  # Confirm image reception

        if t < 0:  # End of data signal
            break

        image = torch.as_tensor(image, device='cuda', dtype=torch.float32).permute(2, 0, 1)

        # Ensure dimensions are divisible by 16 to prevent tensor errors
        H, W = image.shape[1], image.shape[2]
        H = H - (H % 16)
        W = W - (W % 16)
        image = image[:, :H, :W]
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if show_img:
            show_image(image, 1)

        if slam is None:
            print(f"Initializing SLAM with image size: {image.shape[-2]}x{image.shape[-1]}")
            slam = DPVO(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

        end_time = time.time()
        frame_times.append(end_time - start_time)

    reader.join()

    # Calculate `group_times` to contain correct information for every 15 frames
    group_size = 15
    overlap = 5
    group_times = []

    for i in range(0, len(frame_times), group_size - overlap):
        group = frame_times[i:i+group_size]
        if len(group) > 0:
            group_times.append({
                'start_frame': i * stride + start_frame,
                'num_frames': len(group),
                'time_sec': sum(group)
            })

    # Ensure `slam` is not None before terminating
    if slam is not None:
        traj_est, timestamps = slam.terminate()
    else:
        traj_est, timestamps = None, None

    return traj_est, timestamps, group_times  # Return the three values

def modified_plot_trajectory(all_traj_est, traj_ref, titles, save_path):
    """
    Plots the estimated and reference trajectories.
    
    Args:
        all_traj_est (list): List of estimated trajectories.
        traj_ref (PoseTrajectory3D): The reference trajectory.
        titles (list): List of titles for each trajectory segment.
        save_path (PdfPages): The path to save the plot.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the reference trajectory
    ax.plot(traj_ref.positions_xyz[:, 0], 
            traj_ref.positions_xyz[:, 1], 
            traj_ref.positions_xyz[:, 2], 
            'g-', label='Ground Truth', linewidth=2)
    
    # Plot each group of 15 frames with 5-frame overlap
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_traj_est)))
    for i, traj in enumerate(all_traj_est):
        ax.plot(traj.positions_xyz[:, 0], 
                traj.positions_xyz[:, 1], 
                traj.positions_xyz[:, 2], 
                color=colors[i], 
                label=f'Frames {titles[i]}',
                alpha=0.7)
    
    ax.set_title("Continuous Trajectory with 15-Frame Groups (5-Frame Overlap)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path.savefig(fig, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--kittidir', type=Path, default=Path("/media/haneen/New Volume/kitti"))
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--max_images', type=int, default=200)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--sequence', type=str, default="01", help="KITTI sequence number (e.g., 01)")  # Add sequence argument
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    # Use the specified sequence instead of all sequences
    sequence = args.sequence
    groundtruth = args.kittidir / "dataset" / "poses" / f"{sequence}.txt"
    poses_ref_full = file_interface.read_kitti_poses_file(groundtruth)

    if args.start_frame is not None and args.end_frame is not None:
        poses_ref = PoseTrajectory3D(
            positions_xyz=poses_ref_full.positions_xyz[args.start_frame:args.end_frame+1],
            orientations_quat_wxyz=poses_ref_full.orientations_quat_wxyz[args.start_frame:args.end_frame+1],
            timestamps=np.arange(args.start_frame, args.end_frame+1, dtype=np.float64) - args.start_frame
        )
    else:
        poses_ref = poses_ref_full

    print(f"Evaluating KITTI {sequence} with {poses_ref.num_poses // args.stride} poses")
    scene_results = []

    for trial_num in range(args.trials):
        traj_est, timestamps , group_times= run(cfg, args.network, args.kittidir, sequence, args.stride, args.start_frame, args.end_frame, args.viz, args.show_img, args.max_images)
        print(f"\nTrial {trial_num+1} Timing Results:")
        for i, group in enumerate(group_times):
            print(f"Group {i+1}:")
            print(f"Frames: {group['start_frame']}-{group['start_frame']+group['num_frames']}")
            print(f"Number of poses: {group['num_frames']}")
            print(f"Processing time: {group['time_sec']:.2f} sec")
            print(f"Time per frame: {group['time_sec']/group['num_frames']:.3f} sec\n")
        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:, :3],
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
            timestamps=(timestamps * args.stride) - args.start_frame
        )

        traj_ref, traj_est = sync.associate_trajectories(poses_ref, traj_est)
        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        ate_score = result.stats["rmse"]

        if args.plot:
            group_size = 15  # Group size of 15 frames
            overlap = 5  # Overlap of 5 frames between groups
            num_groups = (len(traj_est.positions_xyz) - overlap) // (group_size - overlap)
            Path("trajectory_plots").mkdir(exist_ok=True)
            save_path = f"trajectory_plots/kitti_seq{sequence}_continuous_overlap.pdf"

            all_traj_est_segments = []
            titles = []

            # Create overlapping groups (each group starts from the last frame of the previous group minus 5 frames)
            for group_idx in range(num_groups):
                start_idx = group_idx * (group_size - overlap)
                end_idx = start_idx + group_size
                
                # Ensure not to exceed the end of the trajectory
                if end_idx > len(traj_est.positions_xyz):
                    end_idx = len(traj_est.positions_xyz)
                
                current_traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est.positions_xyz[start_idx:end_idx],
                    orientations_quat_wxyz=traj_est.orientations_quat_wxyz[start_idx:end_idx],
                    timestamps=traj_est.timestamps[start_idx:end_idx]
                )
                all_traj_est_segments.append(current_traj_est)
                titles.append(f"{start_idx}-{end_idx}")

            with PdfPages(save_path) as pdf:
                modified_plot_trajectory(
                    all_traj_est_segments,
                    traj_ref,
                    titles,
                    pdf
                )

        if args.save_trajectory:
            Path("saved_trajectories").mkdir(exist_ok=True)
            file_interface.write_tum_trajectory_file(f"saved_trajectories/KITTI_{sequence}.txt", traj_est)

        scene_results.append(ate_score)

    print(f"Results for sequence {sequence}: {sorted(scene_results)}")
    print(f"Median ATE: {np.median(scene_results)}")