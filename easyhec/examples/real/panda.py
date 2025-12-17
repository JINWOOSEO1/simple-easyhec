import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from transforms3d.euler import euler2mat
from urchin import URDF
import rospy
import os
import pdb
import cv2

from easyhec import ROBOT_DEFINITIONS_DIR
from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from easyhec.utils.utils_3d import merge_meshes
from easyhec.examples.real.data_capture_node import DataCaptureNode

@dataclass
class PandaArgs(Args):
    """Calibrate a (realsense) camera with LeRobot SO100. Note that this script might not work with your particular realsense camera, modify as needed. Other cameras can work if you modify the code to get the camera intrinsics and a single color image from the camera. Results are saved to {output_dir} and organized by the camera name specified in the robot config. Currently only supports off-hand cameras
    
    For your own usage you may have a different camera setup, robot, calibration offsets etc., so we recommend you to copy this file at https://github.com/stonet2000/simple-easyhec/blob/main/easyhec/examples/real/so100.py. 
    
    Before usage make sure to calibrate the robot's motors according to the LeRobot tutorial and look for all comments that start with "CHECK:" which highlight the following:

    1. Check the robot config and make sure the correct camera is used. The default script is for a single realsense camera labelled as "base_camera".
    2. Check and modify the CALIBRATION_OFFSET dictionary to match your own robot's calibration offsets. This is extremely important to tune and is necessary since the 0 degree position of the joints in the real world when calibrated with LeRobot currently do not match the 0 degree position when rendered/simulated.
    3. Modify the initial extrinsic guess if the optimization process fails to converge to a good solution. To save time you can also turn on --use-previous-captures to skip the data collection process if already done once.

    Note that LeRobot SO100 motor calibration is done by moving most joints from one end to another. Make sure to move the joints are far as possible during the LeRobot tutorial on caibration for best results.

    """
    output_dir: str = "results/panda"
    use_previous_captures: bool = False
    """If True, will use the previous collected images and robot segmentations if they exist which can save you time. Otherwise, will prompt you to generate a new segmentation mask. This is useful if you find the initial extrinsic guess is not good enough and simply want to refine that and want to skip the segmentation process."""

    camera_num: int = 3 # total camera number
    camera_idx: int = 1 # which camera to calibrate (1-indexed)



def main(args: PandaArgs):
    rospy.init_node("data_capture_node", anonymous=True)
    capture_node = DataCaptureNode(num_cameras=args.camera_num)

    print(f"Found {args.camera_num} cameras to calibrate")
    for k in range(1, args.camera_num+1):
        (Path(args.output_dir) / 'panda' / f"{k}").mkdir(parents=True, exist_ok=True)
    
    ### Make an initial guess for the extrinsic for each camera ###
    # CHECK: Double check this initial extrinsic guess is roughly close to the real world.
    initial_extrinsic_guesses = dict()
    for k in range(1,args.camera_num+1):
        initial_extrinsic_guess = np.eye(4)

        # the guess says we are at position xyz=[-0.4, 0.0, 0.4] and angle the camerea downwards by np.pi / 4 radians  or 45 degrees
        # note that this convention is more natural for robotics (follows the typical convention for ROS and various simulators), where +Z is moving up towards the sky, +Y is to the left, +X is forward
        initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, +np.pi / 5)
        initial_extrinsic_guess[:3, 3] = np.array([-0.8, 0.2, 1.0])
        initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

        initial_extrinsic_guesses[k] = initial_extrinsic_guess
    
    for k in initial_extrinsic_guesses.keys():
        extrinsics = np.load(f"easyhec/examples/real/calibration_data/cam_{k}_extrinsics.npy")
        gl2cv = np.diag([1, -1, -1, 1])
        # extrinsics[0:3, 3] = extrinsics[0:3, 3] * 1.3  # scale translation to roughly match

        extrinsics = extrinsics @ gl2cv
        extrinsics = np.linalg.inv(extrinsics)

        initial_extrinsic_guesses[k] = extrinsics.copy()

    print("Initial extrinsic guesses")
    for k in initial_extrinsic_guesses.keys():
        print(f"Camera {k}:\n{repr(initial_extrinsic_guesses[k])}")
        

    while capture_node.get_camera_intrinsic(0) is None or capture_node.rgb_images[0] is None:
        rospy.sleep(0.2)

    intrinsics = dict()
    for k in range(1, args.camera_num+1):
        intrinsics[k] = capture_node.get_camera_intrinsic(k-1).copy()
    
    robot_def_path = '/home/seojinwoo/simple-easyhec/easyhec/examples/real/robot_definitions/panda/'
    robot_urdf = URDF.load(os.path.join(robot_def_path,'urdf/panda.urdf'))

    meshes = []
    for link in robot_urdf.links:
        if not link.visuals:
            continue
        if (link.name == 'panda_leftfinger') | (link.name == 'panda_rightfinger'):
            continue
        print(f"Adding link: {link.name}")
        
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        
        meshes.append(merge_meshes(link_meshes))
        
    first_img = capture_node.rgb_images[0].copy()
    H = first_img.shape[0]
    W = first_img.shape[1]
    image_dataset = dict()
    for k in range(1,args.camera_num+1):
        image_dataset[k] = capture_node.rgb_images[k-1].copy()[None]

    # get link poses
    link_poses_dataset = np.zeros((1, 9, 4, 4), dtype = np.float32)
    cfg = dict()

    qpos = capture_node.joint_states
    cfg = {
        'panda_joint1': qpos[0],
        'panda_joint2': qpos[1],
        'panda_joint3': qpos[2],
        'panda_joint4': qpos[3],
        'panda_joint5': qpos[4],
        'panda_joint6': qpos[5],
        'panda_joint7': qpos[6]
    }
    
    link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
    del link_poses['panda_link8']
    del link_poses['panda_hand_tcp']
    del link_poses['panda_leftfinger']
    del link_poses['panda_rightfinger']

    for link_idx, v in enumerate(link_poses.values()):
        link_poses_dataset[0, link_idx] = v

    ### Camera Calibration Process below ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.camera_idx in initial_extrinsic_guesses.keys():
        k = args.camera_idx

        print(f"Calibrating camera {k}")
        initial_extrinsic_guess = initial_extrinsic_guesses[k]
        intrinsic = intrinsics[k]
        images = image_dataset[k]
        camera_mount_poses = None # TODO (stao): support this
        camera_width = images.shape[2]
        camera_height = images.shape[1]
        
        mask_path = Path(args.output_dir) / 'panda' / f"{k}" / f"mask.npy"
        if args.use_previous_captures and mask_path.exists():
            print(f"Using previous mask from {mask_path}")
            masks = np.load(mask_path)
        else:
            interactive_segmentation = InteractiveSegmentation(
                segmentation_model="sam2",
                segmentation_model_cfg=dict(
                    checkpoint=args.checkpoint, model_cfg=args.model_cfg
                ),
            )
            masks = interactive_segmentation.get_segmentation(images)
            np.save(mask_path, masks)

        ### run the optimization given the data ###
        predicted_camera_extrinsic_opencv = (
            optimize(
                camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
                masks=torch.from_numpy(masks).float().to(device),
                link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
                initial_extrinsic_guess=torch.tensor(initial_extrinsic_guess)
                .float()
                .to(device),
                meshes=meshes,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_mount_poses=(
                    torch.from_numpy(camera_mount_poses).float().to(device)
                    if camera_mount_poses is not None
                    else None
                ),
                gt_camera_pose=None,
                iterations=args.train_steps,
                early_stopping_steps=args.early_stopping_steps,
            )
            .cpu()
            .numpy()
        )
        predicted_camera_extrinsic_ros = opencv2ros(predicted_camera_extrinsic_opencv)

        ### Print predicted results ###

        print(f"Predicted camera extrinsic")
        print(f"OpenCV:\n{repr(predicted_camera_extrinsic_opencv)}")
        print(f"ROS/SAPIEN/ManiSkill/Mujoco/Isaac:\n{repr(predicted_camera_extrinsic_ros)}")

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            Path(args.output_dir) / 'panda' / f"{k}" / "camera_extrinsic_opencv.npy",
            predicted_camera_extrinsic_opencv,
        )
        np.save(
            Path(args.output_dir) / 'panda' / f"{k}" / "camera_extrinsic_ros.npy",
            predicted_camera_extrinsic_ros,
        )
        np.save(Path(args.output_dir) / 'panda' / f"{k}" / "camera_intrinsic.npy", intrinsic)

        visualization.visualize_extrinsic_results(
            images=images,
            link_poses_dataset=link_poses_dataset,
            meshes=meshes,
            intrinsic=intrinsic,
            extrinsics=np.stack(
                [initial_extrinsic_guess, predicted_camera_extrinsic_opencv]
            ),
            masks=masks,
            labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
            output_dir=str(Path(args.output_dir) / 'panda' / f"{k}"),
        )
        print(f"Visualizations saved to {Path(args.output_dir)} / 'panda' / {k}")
    else:
        print(f"No initial extrinsic guess found for camera {args.camera_idx}, skipping...")

if __name__ == "__main__":
    main(tyro.cli(PandaArgs))