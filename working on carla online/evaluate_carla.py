from ipaddress import ip_interface
from itertools import count
from multiprocessing import Process, Queue
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

from evo.core import sync
from evo.core import metrics
from evo.core.metrics import PoseRelation
from evo.tools.settings import SETTINGS
from scipy.spatial.transform import Rotation as R  
import argparse  
import sys  
import glob  

# EVO settings for visualization
SETTINGS = {
    "plot_figsize": (10, 6),
    "plot_linewidth": 1.5
}

# Constants
MAX_IMAGES = 500  # Maximum number of images to process
GROUP_SIZE = 15   # Number of frames in each group
OVERLAP = 5       # Overlap between consecutive groups

def show_image(image, t=0):
    """
    Displays an image using OpenCV.

    Args:
        image (torch.Tensor): The image tensor to display.
        t (int): Time in milliseconds to wait for a key event. Default is 0.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('Carla Image', image / 255.0)
    cv2.waitKey(t)

def carla_image_stream(queue, carla_dir, stride=1, start_frame=0, end_frame=None, max_images=500):
    """
    Streams images from the Carla dataset into a queue for processing.

    Args:
        queue (Queue): The queue to store the images.
        carla_dir (Path): The base directory of the Carla dataset.
        stride (int): The stride to use when selecting frames. Default is 1.
        start_frame (int): The starting frame index. Default is 0.
        end_frame (int): The ending frame index. Default is None.
        max_images (int): The maximum number of images to process. Default is 500.
    """
    print("🚀 Starting Carla image stream...")
    images_path = Path(carla_dir) / "frames"
    all_images = sorted(images_path.glob("*.png"))

    if len(all_images) == 0:
        raise FileNotFoundError(f"No images found in {images_path}")

    if end_frame is None or end_frame > len(all_images):
        end_frame = len(all_images)

    image_list = all_images[start_frame:end_frame + 1][::stride][:max_images]
    print(f"🔄 Loaded {len(image_list)} images from Carla")

    # Compute camera intrinsics based on Carla's camera settings
    fov = 90  # Field of view in degrees
    width, height = 640, 480  # Image resolution
    fx = (width / 2) / np.tan(np.radians(fov) / 2)
    fy = fx  # Assume square pixels
    intrinsics = np.array([fx, fy, width / 2, height / 2])

    for t, imfile in enumerate(image_list, start=start_frame):
        image = cv2.imread(str(imfile))
        if image is None:
            print(f"⚠️ Warning: Could not read {imfile}")
            continue

        queue.put((t, image, intrinsics))
        print(f"📤 Sent frame {t} to queue")

    queue.put((-1, None, None))  # Signal end of data
    print("✅ Finished sending images")

@torch.no_grad()
def run_carla(cfg, network, carla_dir, stride=1, start_frame=0, end_frame=None, viz=False, show_img=False, max_images=500):
    """
    Runs the SLAM algorithm on the Carla dataset.

    Args:
        cfg: Configuration object.
        network (str): Path to the network model.
        carla_dir (Path): The base directory of the Carla dataset.
        stride (int): The stride to use when selecting frames. Default is 1.
        start_frame (int): The starting frame index. Default is 0.
        end_frame (int): The ending frame index. Default is None.
        viz (bool): Whether to visualize the process. Default is False.
        show_img (bool): Whether to show images during processing. Default is False.
        max_images (int): The maximum number of images to process. Default is 500.

    Returns:
        tuple: Estimated trajectory, timestamps, and group times.
    """
    torch.backends.cudnn.benchmark = True
    torch.cuda.amp.autocast(enabled=True)

    slam = None
    queue = Queue(maxsize=10)  # Queue to hold images
    frame_times = []

    # Start the image streaming process
    reader = Process(target=carla_image_stream, args=(queue, carla_dir, stride, start_frame, end_frame, max_images))
    reader.start()

    for step in count(start=1):
        start_time = time.time()

        # Wait until images are available in the queue
        while queue.qsize() == 0:
            print("⏳ Waiting for images in queue...")
            time.sleep(0.1)

        (t, image, intrinsics) = queue.get()
        print(f"📥 Received frame {t} from queue")

        if t < 0:  # End of data signal
            break

        # Convert image to PyTorch tensor and move to GPU
        image = torch.as_tensor(image, device='cuda', dtype=torch.float32).permute(2, 0, 1)

        # Ensure image dimensions are divisible by 16
        H, W = image.shape[1], image.shape[2]
        H = H - (H % 16)
        W = W - (W % 16)
        image = image[:, :H, :W]

        # Convert camera intrinsics to PyTorch tensor
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        # Show image if enabled
        if show_img:
            show_image(image, 1)

        # Initialize SLAM if not already initialized
        if slam is None:
            print(f"Initializing SLAM with image size: {image.shape[-2]}x{image.shape[-1]}")
            slam = DPVO(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)

        # Process image using SLAM
        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

        # Record processing time
        end_time = time.time()
        frame_times.append(end_time - start_time)

    reader.join()

    # Calculate group times for every 15 frames with 5-frame overlap
    group_times = []
    for i in range(0, len(frame_times), GROUP_SIZE - OVERLAP):
        group = frame_times[i:i + GROUP_SIZE]
        if len(group) > 0:
            group_times.append({
                'start_frame': i * stride + start_frame,
                'num_frames': len(group),
                'time_sec': sum(group)
            })

    # Terminate SLAM and retrieve estimated trajectory
    if slam is not None:
        traj_est, timestamps = slam.terminate()
    else:
        traj_est, timestamps = None, None

    return traj_est, timestamps, group_times

def load_ground_truth(ground_truth_path, start_frame=0, end_frame=None):
    """
    Loads ground truth poses from a text file.

    Args:
        ground_truth_path (Path): Path to the ground truth file.
        start_frame (int): Starting frame index. Default is 0.
        end_frame (int): Ending frame index. Default is None.

    Returns:
        np.array: Ground truth poses.
    """
    poses = []
    with open(ground_truth_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            frame_id = int(data[0])
            x, y, z = float(data[1]), float(data[2]), float(data[3])
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
            poses.append([x, y, z, qw, qx, qy, qz])

    poses = np.array(poses)
    if end_frame is not None:
        poses = poses[start_frame:end_frame + 1]
    else:
        poses = poses[start_frame:]

    return poses

def save_poses_to_pdf(poses, save_path):
    """
    Saves poses to a PDF file as a table.

    Args:
        poses (np.array): The poses to save.
        save_path (str): Path to save the PDF file.
    """
    with PdfPages(save_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        if poses.shape[1] == 3:  # If data contains 3 columns (x, y, z)
            for i, pose in enumerate(poses):
                table_data.append([i, pose[0], pose[1], pose[2]])
            col_labels = ["Frame", "X", "Y", "Z"]
        elif poses.shape[1] == 7:  # If data contains 7 columns (x, y, z, qw, qx, qy, qz)
            for i, pose in enumerate(poses):
                table_data.append([i, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]])
            col_labels = ["Frame", "X", "Y", "Z", "qw", "qx", "qy", "qz"]
        else:
            raise ValueError("Poses must have 3 or 7 columns.")

        # Create the table
        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def plot_grouped_trajectories(traj_est_groups, traj_ref_groups, save_path):
    """
    Plots all trajectory groups on the same plot with different colors.

    Args:
        traj_est_groups (list): List of estimated trajectory groups.
        traj_ref_groups (list): List of ground truth trajectory groups.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(14, 10), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(traj_est_groups)))

    for i, (traj_est, traj_ref) in enumerate(zip(traj_est_groups, traj_ref_groups)):
        # Plot ground truth
        plt.plot(traj_ref[:, 0], traj_ref[:, 1], color=colors[i], linestyle='--', linewidth=1, label=f'Ground Truth (Group {i+1})')
        # Plot estimated trajectory
        plt.plot(traj_est[:, 0], traj_est[:, 1], color=colors[i], linestyle='-', linewidth=2, label=f'Estimated Trajectory (Group {i+1})')

    plt.title("2D Trajectory Comparison (All Groups)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True)

    # Save the plot to a PDF file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--carla_dir', type=Path, default=Path.home() / "DPVO" / "carlaData")
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--ground_truth_path', type=Path, required=True, help="Path to the ground truth file")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    # Load ground truth
    poses_ref = load_ground_truth(args.ground_truth_path, args.start_frame, args.end_frame)
    poses_ref = PoseTrajectory3D(
        positions_xyz=poses_ref[:, :3],
        orientations_quat_wxyz=poses_ref[:, [3, 4, 5, 6]],
        timestamps=np.arange(len(poses_ref), dtype=np.float64)
    )

    scene_results = []
    for trial_num in range(args.trials):
        traj_est, timestamps, group_times = run_carla(cfg, args.network, args.carla_dir, args.stride, args.start_frame, args.end_frame, args.viz, args.show_img, args.max_images)
        print(f"\nTrial {trial_num+1} Timing Results:")
        for i, group in enumerate(group_times):
            print(f"Group {i+1}:")
            print(f"Frames: {group['start_frame']}-{group['start_frame']+group['num_frames']}")
            print(f"Number of poses: {group['num_frames']}")
            print(f"Processing time: {group['time_sec']:.2f} sec")
            print(f"Time per frame: {group['time_sec']/group['num_frames']:.3f} sec\n")

        # Convert estimated trajectory to PoseTrajectory3D
        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:, :3],  # x, y, z
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],  # qw, qx, qy, qz
            timestamps=timestamps
        )

        # Synchronize trajectories
        traj_ref, traj_est = sync.associate_trajectories(poses_ref, traj_est)

        # Calculate ATE
        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        ate_score = result.stats["rmse"]

        if args.plot:
            group_size = 15
            overlap = 5
            num_groups = (len(traj_est.positions_xyz) - overlap) // (group_size - overlap)

            traj_est_groups = []
            traj_ref_groups = []

            for group_idx in range(num_groups):
                start_idx = group_idx * (group_size - overlap)
                end_idx = start_idx + group_size

                if end_idx > len(traj_est.positions_xyz):
                    end_idx = len(traj_est.positions_xyz)

                traj_est_group = traj_est.positions_xyz[start_idx:end_idx]
                traj_ref_group = traj_ref.positions_xyz[start_idx:end_idx]

                if len(traj_est_group) == 0 or len(traj_ref_group) == 0:
                    print(f"⚠️ Group {group_idx+1} is empty. Skipping...")
                    continue

                traj_est_groups.append(traj_est_group)
                traj_ref_groups.append(traj_ref_group)

            if len(traj_est_groups) == 0:
                print("❌ No valid groups to plot!")
            else:
                Path("trajectory_plots").mkdir(exist_ok=True)
                save_path = "trajectory_plots/carla_grouped_trajectories.pdf"
                plot_grouped_trajectories(traj_est_groups, traj_ref_groups, save_path)
                print("✅ Trajectories plotted successfully.")

        if args.save_trajectory:
            Path("saved_trajectories").mkdir(exist_ok=True)
            np.savetxt(f"saved_trajectories/CARLA.txt", traj_est.positions_xyz, fmt='%.6f')

        scene_results.append(ate_score)
        # Save poses to PDF
        Path("poses_pdf").mkdir(exist_ok=True)
        save_poses_to_pdf(traj_est.positions_xyz, "poses_pdf/carla_poses.pdf")
        print("✅ Saved poses to PDF.")

    print(f"Median ATE: {np.median(scene_results)}")
    print("✅ Processing completed successfully.")