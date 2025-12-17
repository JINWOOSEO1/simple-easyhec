import os
from argparse import ArgumentParser
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo, JointState
from franka_msgs.msg import FrankaState
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

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


class CalibrationNode:
    def __init__(self, cam_idx=1, save_dir="calibration_data", visualize=False):
        self.cam_idx = cam_idx
        self.save_dir = save_dir
        self.visualize = visualize

        rgb_topic = f"/cam_{cam_idx}/rgb/compressed"
        camera_info_topic = f"/cam_{cam_idx}/rgb/camera_info"
        franka_state_topic = "/franka_state_controller/franka_states"

        self.K = None

        rospy.Subscriber(rgb_topic, CompressedImage, self.rgb_callback)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(franka_state_topic, FrankaState, self.franka_state_callback)

    def rgb_callback(self, msg):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_COLOR)
        self.rgb_image = cv_image.astype(np.uint8)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)

    def franka_state_callback(self, msg: FrankaState):
        self.ee_pose = np.array(msg.O_T_EE).reshape(4, 4).T

    def run(self):
        extrinsic_file = os.path.join(
            self.save_dir, f"cam_{self.cam_idx}_extrinsics.npy"
        )
        extrinsics = self.calibrate(visualize=self.visualize)
        if extrinsics is None:
            print("Calibration failed.")
            return

        np.save(extrinsic_file, extrinsics)

        print(f"Saved extrinsics to {extrinsic_file}")

    def calibrate(self, visualize=False):
        while (
            self.K is None
            or not hasattr(self, "rgb_image")
            or not hasattr(self, "ee_pose")
        ):
            print("Waiting for data...")
            rospy.sleep(0.1)

        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []

        while not rospy.is_shutdown():
            if not hasattr(self, "rgb_image") or self.rgb_image is None:
                rospy.sleep(0.1)
                continue

            if self.K is None or self.ee_pose is None:
                rospy.sleep(0.1)
                continue

            rgb_image = self.rgb_image.copy()

            cv2.imshow("Calibration", rgb_image)
            key = cv2.waitKey(1)

            if key == ord("q"):
                print("Exiting calibration.")
                break
            elif key == ord("c"):
                ee_pose = self.ee_pose.copy()
                K = self.K.copy()

                # 30 mm ArUco marker
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejectedImgPoints = detector.detectMarkers(rgb_image)
                if ids is None or len(ids) == 0:
                    print("No markers detected, please adjust the camera.")
                    continue

                if len(ids) > 1:
                    print(
                        "Multiple markers detected, please ensure only one is visible."
                    )
                    continue

                vis = rgb_image.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imshow("Detected Markers", vis)

                marker_length = 0.17  # 170 mm
                half_size = marker_length / 2.0
                obj_points = np.array(
                    [
                        [-half_size, half_size, 0],
                        [half_size, half_size, 0],
                        [half_size, -half_size, 0],
                        [-half_size, -half_size, 0],
                    ],
                    dtype=np.float32,
                )

                image_points = corners[0][0]
                success, rvec, tvec = cv2.solvePnP(obj_points, image_points, K, None)

                R_target2cam_i, _ = cv2.Rodrigues(rvec)
                t_target2cam_i = tvec.reshape(3)

                R_gripper2base_i = ee_pose[:3, :3]
                t_gripper2base_i = ee_pose[:3, 3]

                R_gripper2base.append(R_gripper2base_i)
                t_gripper2base.append(t_gripper2base_i)
                R_target2cam.append(R_target2cam_i)
                t_target2cam.append(t_target2cam_i)
                print(f"Collected {len(R_gripper2base)} samples.")

        R_gripper2cam, t_gripper2cam = self.calibrate_eye_hand(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True
        )

        T_gripper2cam = np.eye(4)
        T_gripper2cam[:3, :3] = R_gripper2cam
        T_gripper2cam[:3, 3] = t_gripper2cam[:, 0]

        gl2cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        T_gripper2cam = T_gripper2cam @ gl2cv

        return T_gripper2cam

    def calibrate_eye_hand(
        self,
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        eye_to_hand=True,
    ):
        if eye_to_hand:
            # change coordinates from gripper2base to base2gripper
            R_base2gripper, t_base2gripper = [], []
            for R, t in zip(R_gripper2base, t_gripper2base):
                R_b2g = R.T
                t_b2g = -R_b2g @ t
                R_base2gripper.append(R_b2g)
                t_base2gripper.append(t_b2g)

            # change parameters values
            R_gripper2base = R_base2gripper
            t_gripper2base = t_base2gripper

        # calibrate
        R, t = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
        )

        return R, t


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cam_idx", type=int, default=1, help="Camera index to use")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="calibration_data",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the calibration process",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    rospy.init_node("calibration_node", anonymous=True)

    capture_node = CalibrationNode(
        cam_idx=args.cam_idx, save_dir=args.save_dir, visualize=args.visualize
    )
    capture_node.run()
