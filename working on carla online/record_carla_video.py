import carla
import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

# إعداد اتصال CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# اختيار السيارة والكاميرا من مكتبة CARLA
blueprint_library = world.get_blueprint_library()
car_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(car_bp, spawn_point)

# تفعيل القيادة الذاتية (Autopilot)
vehicle.set_autopilot(True)

# إعداد الكاميرا
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# تحديد المسارات لحفظ البيانات
base_path = "/home/haneen/DPVO/carlaData"
frames_folder = os.path.join(base_path, "frames")
ground_truth_folder = os.path.join(base_path, "ground_truth")
os.makedirs(base_path, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)

video_path = os.path.join(base_path, "carla_video.avi")
ground_truth_path = os.path.join(ground_truth_folder, "ground_truth.txt")

# إعداد تسجيل الفيديو
frame_width, frame_height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

# متغير لتتبع عدد الفريمات
frame_count = 0

# دالة لحفظ الصور والفيديو
def process_image(image):
    global frame_count

    # تحويل الصورة من بيانات CARLA إلى مصفوفة NumPy
    img_data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    
    # حفظ الصورة كفريم منفصل
    frame_filename = os.path.join(frames_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, img_data)

    # حفظ الفريم في الفيديو
    out.write(img_data)

    # تحديث عدد الفريمات
    frame_count += 1

# دالة لحفظ Ground Truth (الإحداثيات العالمية والاتجاهات والطابع الزمني)
def save_ground_truth():
    global frame_count

    # استخراج إحداثيات السيارة في النظام العالمي
    transform = vehicle.get_transform()
    location = transform.location
    x, y, z = location.x, location.y, location.z

    # استخراج الاتجاهات وتحويلها إلى Quaternion
    rotation = transform.rotation
    roll, pitch, yaw = rotation.roll, rotation.pitch, rotation.yaw
    quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()  # تحويل إلى [qx, qy, qz, qw]

    # الحصول على الطابع الزمني الحالي
    timestamp = time.time()

    # حفظ البيانات في ملف Ground Truth
    with open(ground_truth_path, 'a') as f:
        f.write(f"{frame_count} {x} {y} {z} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {timestamp}\n")

# تفعيل الكاميرا
camera.listen(lambda image: process_image(image))

# تشغيل المحاكي لمدة 30 ثانية
start_time = time.time()
while time.time() - start_time < 30:
    save_ground_truth()  # حفظ Ground Truth في كل إطار
    time.sleep(0.05)  # انتظر 50 مللي ثانية بين كل إطار

# إيقاف التسجيل والتدمير
camera.stop()
out.release()

# تدمير السيارة بعد انتهاء التسجيل
print("🛑 إيقاف السيارة وحذفها من المحاكي...")
camera.destroy()
vehicle.destroy()

print(f"✅ تم حفظ الفيديو في: {video_path}")
print(f"✅ تم حفظ الفريمات في: {frames_folder}")
print(f"✅ تم حفظ Ground Truth في: {ground_truth_path}")