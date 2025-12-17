from argparse import ArgumentParser
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo, JointState
from threading import Lock

def ros_compressed_image_to_cv_image(
    msg: CompressedImage, encoding=cv2.IMREAD_UNCHANGED
):
    # Decompress the image data, jpeg byte array to cv2 image
    # np_arr = np.frombuffer(msg.data, np.uint8)
    np_arr = np.fromstring(msg.data, np.uint8)
    if np_arr.size == 0:
        rospy.logerr("Received empty image data")
        return None
    cv_image = cv2.imdecode(np_arr, encoding)
    return cv_image

class DataCaptureNode:
    def __init__(self, num_cameras=1, crop_regions_path=None, filter_depth=False):
        self.num_cameras = num_cameras
        self.filter_depth = filter_depth

        self.rgb_subscribers = []
        self.depth_subscribers = []
        self.mask_subscribers = []
        self.camera_info_subscribers = []
        self.state_subscribers = []

        self.rgb_images = [None] * num_cameras
        self.depth_images = [None] * num_cameras
        self.mask_images = [None] * num_cameras
        self.camera_infos = [None] * num_cameras
        self.joint_states = [None] * 7 # 7dof
        
        self.lock = Lock()

        self.ee_pose = None
        self.crop_regions = None
        if crop_regions_path is not None:
            self.crop_regions = np.load(crop_regions_path, allow_pickle=True)

        for i in range(num_cameras):
            rgb_topic = f"/cam_{i+1}/rgb/compressed"
            depth_topic = f"/cam_{i+1}/depth/compressed"
            mask_topic = f"/cam_{i+1}/mask/compressed"
            camera_info_topic = f"/cam_{i+1}/rgb/camera_info"
            state_topic = f"/franka_state_controller/joint_states"

            self.rgb_subscribers.append(
                rospy.Subscriber(
                    rgb_topic, CompressedImage, self.rgb_callback, callback_args=i
                )
            )
            self.depth_subscribers.append(
                rospy.Subscriber(
                    depth_topic, CompressedImage, self.depth_callback, callback_args=i
                )
            )
            self.mask_subscribers.append(
                rospy.Subscriber(
                    mask_topic, CompressedImage, self.mask_callback, callback_args=i
                )
            )
            self.camera_info_subscribers.append(
                rospy.Subscriber(
                    camera_info_topic,
                    CameraInfo,
                    self.camera_info_callback,
                    callback_args=i,
                )
            )
            self.state_subscribers.append(
                rospy.Subscriber(
                    state_topic, JointState, self.state_callback,
                )
            )

    def rgb_callback(self, msg, camera_index):
        # cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_COLOR).copy()
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED).copy()

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        with self.lock:
            self.rgb_images[camera_index] = cv_image.astype(np.uint8)
        
    def mask_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED).copy()

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        with self.lock:
            self.mask_images[camera_index] = cv_image.astype(np.uint8)

    def depth_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED).copy()

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]
        
        if self.filter_depth:
            # apply surface normal filtering
            cv_image = self._filter_depth(cv_image, filter_threshold=0.01, depth_scale=0.001)

        with self.lock:
            self.depth_images[camera_index] = cv_image.copy()
    
    def state_callback(self, msg: JointState):
        self.joint_states = msg.position

    def _filter_depth(
        self,
        depth_image: np.ndarray, 
        filter_threshold: float = 0.5,
        depth_scale: float = 1.0, 
    ) -> np.ndarray:
        """
        Calculates approximate normals in image space using depth gradient 
        and filters out surfaces perpendicular to the camera (Z-axis).
        This method is fast but inaccurate in 3D space as it ignores camera intrinsics (K).

        Args:
            depth_image (np.ndarray): H x W depth map (e.g., in mm or scaled).
            filter_threshold (float): Threshold for gradient magnitude. 
                                    If |(gu, gv)| < threshold, the surface is considered 
                                    perpendicular to the camera and filtered (set to 0).
            depth_scale (float): Scale factor to convert depth units to meters (e.g., 0.001).

        Returns:
            np.ndarray: H x W array of filtered depths
        """
        # Convert depth to working scale
        D = depth_image.copy().astype(np.float64) * depth_scale

        # 1. Calculate 2D depth gradient (gu: x-dir, gv: y-dir)
        # np.gradient uses central difference for fast calculation.
        gv, gu = np.gradient(D)
        
        # Calculate magnitude of 2D gradient G = (gu, gv)
        gradient_magnitude = np.linalg.norm(np.stack([gu, gv], axis=-1), axis=-1)
        
        filter_mask_3d = gradient_magnitude > filter_threshold
        res = depth_image.copy()
        res[filter_mask_3d] = 0.0  # Set filtered depths to 0

        return res

    def camera_info_callback(self, msg: CameraInfo, camera_index):
        if self.camera_infos[camera_index] is None:
            K = list(msg.K)
            cx = K[2]
            cy = K[5]
            if self.crop_regions is not None:
                x1, y1, x2, y2 = self.crop_regions[camera_index]
                cx -= x1
                cy -= y1
                K[2] = cx
                K[5] = cy
                msg.K = K

            self.camera_infos[camera_index] = msg

    def get_camera_intrinsic(self, index):
        if self.camera_infos[index] is None:
            return None
        return np.array(self.camera_infos[index].K).reshape(3, 3)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_cameras", type=int, default=1, help="Number of cameras to subscribe to"
    )
    parser.add_argument(
        "--crop_regions_path",
        type=str,
        default=None,
        help="Path to load the cropping regions",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Path to save the rgbd stream"
    )
    args = parser.parse_args()

    rospy.init_node("data_capture_node", anonymous=True)

    capture_node = DataCaptureNode(
        num_cameras=args.num_cameras, crop_regions_path=args.crop_regions_path, filter_depth=True
    )

    if args.save_dir is not None:
        import os

        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

        for i in range(args.num_cameras):
            os.makedirs(os.path.join(args.save_dir, f"{i+1}"), exist_ok=True)

    img_idx = 0
    is_recording = False
    while not rospy.is_shutdown():
        for i in range(args.num_cameras):
            if capture_node.rgb_images[i] is not None:
                cv2.imshow(f"Camera {i} RGB", capture_node.rgb_images[i])
            # if capture_node.rgb_right_images[i] is not None:
            #     cv2.imshow(f"Camera {i} RGB Right", capture_node.rgb_right_images[i])
            if capture_node.depth_images[i] is not None:
                depth = capture_node.depth_images[i].copy()
                depth = depth / 1000.0

                depth_vis = cv2.convertScaleAbs(depth, alpha=255.0)

                cv2.imshow(f"Camera {i} Depth", depth_vis)

            if args.save_dir is not None and is_recording:
                rgb_image = capture_node.rgb_images[i].copy()
                depth = capture_node.depth_images[i].copy()
                depth = depth / 1000.0

                cv2.imwrite(
                    os.path.join(args.save_dir, f"{i+1}", f"rgb_{img_idx:05d}.png"),
                    rgb_image,
                )
                np.save(
                    os.path.join(args.save_dir, f"{i+1}", f"depth_{img_idx:05d}.npy"),
                    depth,
                )
                np.save(
                    os.path.join(args.save_dir, f"{i+1}", f"intrinsic.npy"),
                    capture_node.get_camera_intrinsic(i),
                )

                if i == args.num_cameras - 1:
                    img_idx += 1

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            is_recording = not is_recording
            print(f"Recording: {is_recording}")