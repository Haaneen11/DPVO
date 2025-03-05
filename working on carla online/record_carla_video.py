import carla
import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

# Set up CARLA connection
client = carla.Client('localhost', 2000)  # Connect to the CARLA server
client.set_timeout(10.0)  # Set timeout for connection
world = client.get_world()  # Get the CARLA world

# Select the vehicle and camera from the CARLA blueprint library
blueprint_library = world.get_blueprint_library()
car_bp = blueprint_library.filter('vehicle.tesla.model3')[0]  # Tesla Model 3 blueprint
spawn_point = world.get_map().get_spawn_points()[0]  # Get a spawn point
vehicle = world.spawn_actor(car_bp, spawn_point)  # Spawn the vehicle

# Enable autopilot mode
vehicle.set_autopilot(True)

# Set up the camera
camera_bp = blueprint_library.find('sensor.camera.rgb')  # RGB camera blueprint
camera_bp.set_attribute('image_size_x', '640')  # Set image width
camera_bp.set_attribute('image_size_y', '480')  # Set image height
camera_bp.set_attribute('fov', '90')  # Set field of view
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # Camera position relative to the vehicle
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)  # Spawn the camera

# Define paths for saving data
base_path = "/home/haneen/DPVO/carlaData"  # Base directory for saving data
frames_folder = os.path.join(base_path, "frames")  # Folder to save individual frames
ground_truth_folder = os.path.join(base_path, "ground_truth")  # Folder to save ground truth data
os.makedirs(base_path, exist_ok=True)  # Create base directory if it doesn't exist
os.makedirs(frames_folder, exist_ok=True)  # Create frames directory if it doesn't exist
os.makedirs(ground_truth_folder, exist_ok=True)  # Create ground truth directory if it doesn't exist

video_path = os.path.join(base_path, "carla_video.avi")  # Path to save the video
ground_truth_path = os.path.join(ground_truth_folder, "ground_truth.txt")  # Path to save ground truth data

# Set up video recording
frame_width, frame_height = 640, 480  # Video frame dimensions
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))  # Video writer object

# Variable to track frame count
frame_count = 0

def process_image(image):
    """
    Processes the image captured by the camera.

    Args:
        image: The raw image data from the CARLA camera.
    """
    global frame_count

    # Convert CARLA image data to a NumPy array
    img_data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

    # Save the image as a separate frame
    frame_filename = os.path.join(frames_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, img_data)

    # Save the frame to the video
    out.write(img_data)

    # Increment frame count
    frame_count += 1

def save_ground_truth():
    """
    Saves the ground truth data (vehicle position, orientation, and timestamp).
    """
    global frame_count

    # Get the vehicle's global transform
    transform = vehicle.get_transform()
    location = transform.location
    x, y, z = location.x, location.y, location.z  # Vehicle position

    # Get the vehicle's orientation and convert it to a quaternion
    rotation = transform.rotation
    roll, pitch, yaw = rotation.roll, rotation.pitch, rotation.yaw
    quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()  # Convert to [qx, qy, qz, qw]

    # Get the current timestamp
    timestamp = time.time()

    # Save the ground truth data to a file
    with open(ground_truth_path, 'a') as f:
        f.write(f"{frame_count} {x} {y} {z} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {timestamp}\n")

# Activate the camera
camera.listen(lambda image: process_image(image))

# Run the simulation for 30 seconds
start_time = time.time()
while time.time() - start_time < 30:
    save_ground_truth()  # Save ground truth for each frame
    time.sleep(0.05)  # Wait 50 milliseconds between frames

# Stop recording and clean up
camera.stop()  # Stop the camera
out.release()  # Release the video writer

# Destroy the vehicle and camera after recording
print("🛑 Stopping the vehicle and removing it from the simulation...")
camera.destroy()  # Destroy the camera
vehicle.destroy()  # Destroy the vehicle

print(f"✅ Video saved to: {video_path}")
print(f"✅ Frames saved to: {frames_folder}")
print(f"✅ Ground truth saved to: {ground_truth_path}")
