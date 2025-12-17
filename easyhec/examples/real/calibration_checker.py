import os
from argparse import ArgumentParser
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from scipy.spatial.transform import Rotation as R
import open3d as o3d

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


class CalibrationChecker:
    def __init__(self, num_cameras=1, extrinsic_dir=None):
        self.num_cameras = num_cameras
        self.extrinsic_dir = extrinsic_dir

        for cam_idx in range(num_cameras):
            rgb_topic = f"/cam_{cam_idx}/rgb/compressed"
            depth_topic = f"/cam_{cam_idx}/depth/compressed"
            camera_info_topic = f"/cam_{cam_idx}/rgb/camera_info"

        self.rgb_images = [None] * self.num_cameras
        self.depth_images = [None] * self.num_cameras
        self.K = [None] * self.num_cameras
        for i in range(num_cameras):
            rgb_topic = f"/cam_{i+1}/rgb/compressed"
            depth_topic = f"/cam_{i+1}/depth/compressed"
            camera_info_topic = f"/cam_{i+1}/rgb/camera_info"

            rospy.Subscriber(
                rgb_topic, CompressedImage, self.rgb_callback, callback_args=i
            )
            rospy.Subscriber(
                depth_topic, CompressedImage, self.depth_callback, callback_args=i
            )
            rospy.Subscriber(
                camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                callback_args=i,
            )

        self.depth_scale = 1000.0

    def rgb_callback(self, msg, cam_idx=0):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_COLOR)
        self.rgb_images[cam_idx] = cv_image.astype(np.uint8)

    def depth_callback(self, msg, cam_idx=0):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED)
        depth_image = cv_image.astype(np.uint16)
        self.depth_images[cam_idx] = (
            # depth_image.astype(np.float32) / 1000.0
            depth_image.astype(np.float32)
            / self.depth_scale
        )  # Convert mm to meters

    def camera_info_callback(self, msg, cam_idx=0):
        if self.K[cam_idx] is None:
            self.K[cam_idx] = np.array(msg.K).reshape(3, 3)

    def run(self):
        # extrinsics = np.load(self.extrinsic_path)
        gl2cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        extrinsics_mv = []
        for cam_idx in range(self.num_cameras):
            extrinsic_path = os.path.join(
                # self.extrinsic_dir, f"cam_{cam_idx+1}_extrinsics.npy"
                self.extrinsic_dir, f"{cam_idx + 1}", "camera_extrinsic_opencv.npy"
            )
            extrinsic = np.load(extrinsic_path)
            extrinsic = np.linalg.inv(extrinsic)
            extrinsics_mv.append(extrinsic)

        # extrinsics_mv = extrinsics_mv @ gl2cv
        # extrinsics_mv = gl2cv @ extrinsics_mv 

        while (
            any(rgb_image is None for rgb_image in self.rgb_images)
            or any(depth_image is None for depth_image in self.depth_images)
            or any(K is None for K in self.K)
        ):
            print("Waiting for data...")
            rospy.sleep(0.1)
        # project depth to point cloud
        vis = o3d.visualization.Visualizer()

        vis.create_window()
        is_init_vis = False

        for cam_idx in range(self.num_cameras):
            self.K[cam_idx][0, 0] *= 0.5
            self.K[cam_idx][1, 1] *= 0.5
            self.K[cam_idx][0, 2] *= 0.5
            self.K[cam_idx][1, 2] *= 0.5

        while not rospy.is_shutdown():
            if not vis.poll_events():
                break

            points_mv = []
            colors_mv = []

            for cam_idx in range(self.num_cameras):
                h, w = self.depth_images[cam_idx].shape
                depth_image = self.depth_images[cam_idx].copy()
                rgb_image = self.rgb_images[cam_idx].copy()
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                K = self.K[cam_idx]
                extrinsics = extrinsics_mv[cam_idx]

                # half resolution
                depth_image = cv2.resize(
                    depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST
                )
                rgb_image = cv2.resize(rgb_image, (w // 2, h // 2))
                h, w = depth_image.shape

                i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
                valid = (depth_image > 0) & (depth_image < 3.0)
                z = np.where(valid, depth_image, np.nan)
                x = np.where(valid, z * (i - K[0, 2]) / K[0, 0], 0)
                y = np.where(valid, z * (j - K[1, 2]) / K[1, 1], 0)
                points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
                colors = rgb_image.reshape(-1, 3) / 255.0
                mask = valid.reshape(-1)
                points = points[mask]
                colors = colors[mask]

                points = extrinsics[:3, :3] @ points.T
                points = points.T + extrinsics[:3, 3]

                workspace = np.array([[-1, 2], [-0.8, 1], [-0.3, 2.0]])
                valid = (
                    (points[:, 0] > workspace[0, 0])
                    & (points[:, 0] < workspace[0, 1])
                    & (points[:, 1] > workspace[1, 0])
                    & (points[:, 1] < workspace[1, 1])
                    & (points[:, 2] > workspace[2, 0])
                    & (points[:, 2] < workspace[2, 1])
                )
                points = points[valid]
                colors = colors[valid]

                points_mv.append(points)
                colors_mv.append(colors)

            points = np.concatenate(points_mv, axis=0)
            colors = np.concatenate(colors_mv, axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd.transform(extrinsics)

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            vis.clear_geometries()
            vis.add_geometry(pcd, reset_bounding_box=not is_init_vis)
            vis.add_geometry(axis, reset_bounding_box=not is_init_vis)
            is_init_vis = True

            vis.update_renderer()

            cv2.imshow(
                "Calibration Checker", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            )
            key = cv2.waitKey(1) & 0xFF
            # up arrow
            # if key == 82:
            #     self.depth_scale += 10.0
            #     print(f"Depth scale: {self.depth_scale}")
            # # down arrow
            # elif key == 84:
            #     self.depth_scale -= 10.0
            #     print(f"Depth scale: {self.depth_scale}")

        # o3d.visualization.draw_geometries([pcd, axis])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_cameras", type=int, default=1, help="Number of cameras to use"
    )
    parser.add_argument(
        "--extrinsic_dir", type=str, required=True, help="Path to save extrinsics"
    )
    args = parser.parse_args()

    rospy.init_node("calibration_checker", anonymous=True)

    checker = CalibrationChecker(
        num_cameras=args.num_cameras, extrinsic_dir=args.extrinsic_dir
    )
    checker.run()
