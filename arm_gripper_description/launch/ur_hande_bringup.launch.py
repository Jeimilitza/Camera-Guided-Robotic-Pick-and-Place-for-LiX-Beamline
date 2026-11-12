#!/usr/bin/env python3
"""
Combined bringup: UR arm driver (ros2_control) + Robotiq Hand-E gripper
controller + MoveIt 2 — so the arm's joints and the gripper can be commanded
simultaneously from the SAME controller_manager.

Usage:
    ros2 launch <your_package> ur_hande_bringup.launch.py robot_ip:=192.167.1.100
    ros2 launch <your_package> ur_hande_bringup.launch.py use_mock_hardware:=true

--------------------------------------------------------------------------
IMPORTANT — three things this file assumes you already have, matching the
pattern in the LiX template. Stock ur_description / robotiq_hande_description
do NOT ship these; you'll need to create them (or point the args at ones
you already have):

  1. description_package / description_file
     A COMBINED xacro (arm + Hand-E) with ONE <ros2_control> tag declaring
     both hardware interfaces. This is what makes it possible for a single
     controller_manager to own both — you cannot get simultaneous control
     out of two separate ros2_control_node processes.
     robotiq_hande_description's own urdf.xacro becomes something you
     xacro:include INTO this combined file, rather than launching stand-alone
     (the robotiq_hande_description "display" launch with joint_state_publisher_gui
     is only useful for eyeballing the gripper model — it fakes joint states
     and isn't part of real hardware control, so it's intentionally not
     wired in here).

  2. controllers_file
     A single ros2_control controllers YAML with the usual UR controllers
     PLUS a gripper_action_controller entry (type + joint name(s) for
     Hand-E). ros2_control_node loads this once at startup, so the gripper
     spawner below only needs to activate the controller by name — it does
     NOT need its own --param-file.

  3. (Not templated below, but you'll want it for MoveIt to actually move the
     gripper) — a MoveIt controllers YAML and SRDF planning group / end
     effector for the gripper. Point .trajectory_execution() /
     .robot_description_semantic() at your moveit_config_package's files.
--------------------------------------------------------------------------
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    tf_prefix = LaunchConfiguration("tf_prefix")
    launch_rviz = LaunchConfiguration("launch_rviz")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    headless_mode = LaunchConfiguration("headless_mode")

    description_share = FindPackageShare(description_package)
    moveit_share = FindPackageShare(moveit_config_package)

    # ur_robot_driver (Jazzy) expects an ABSOLUTE path for description_file —
    # it no longer joins description_package + "urdf" + description_file
    # internally, so we resolve that full path ourselves and pass it in whole.
    description_file_path = PathJoinSubstitution(
        [description_share, "urdf", description_file]
    )

    # ---- 1. UR arm driver -------------------------------------------------
    # Starts ros2_control_node + the arm's controllers. The gripper spawner
    # below activates a controller inside this SAME controller_manager, which
    # is only possible because description_file is the combined arm+gripper
    # xacro (see note above) and controllers_file lists gripper_action_controller
    # alongside the arm's controllers.
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "tf_prefix": tf_prefix,
            "use_mock_hardware": use_mock_hardware,
            "launch_rviz": "false",  # one shared RViz is launched below, with MoveIt
            "description_package": description_package,
            "description_file": description_file_path,
            "headless_mode": headless_mode,
            "controllers_file": controllers_file,
        }.items(),
    )

    # ---- 2. Gripper controller spawner ------------------------------------
    # This is the gripper's "driver" from ROS's point of view: it activates
    # the ros2_control controller for Hand-E in the controller_manager the UR
    # driver just started, so arm and gripper are controlled through the same
    # /controller_manager and can move together. No --param-file needed here
    # since ros2_control_node already loaded gripper_action_controller's
    # params from controllers_file above.
    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_action_controller",
            "-c", "/controller_manager",
        ],
    )

    # ---- 3. MoveIt 2 -------------------------------------------------------
    moveit_config = (
        MoveItConfigsBuilder(
            "ur5e_with_hande",
            package_name="arm_gripper_description",
        )
        .robot_description(
            file_path=os.path.join(
                description_share.perform(context),
                "urdf",
                description_file.perform(context),
            ),
            mappings={
                "name": ur_type.perform(context),
                "ur_type": ur_type.perform(context),
            },
        )
        .robot_description_semantic(
            file_path=os.path.join(
                description_share.perform(context),
                "urdf",
                "ur.srdf.xacro",
            )
        )
        .trajectory_execution(
            file_path=os.path.join(
                description_share.perform(context),
                "config",
                "moveit_controllers.yaml",
            )
        )
        .robot_description_kinematics(
            file_path=os.path.join(
                description_share.perform(context),
                "config",
                "kinematics.yaml",
            )
        )
        .planning_pipelines(
            pipelines=["ompl"],
            default_planning_pipeline="ompl",
            load_all=False,
        )
        .to_moveit_configs()
    )


    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config_file = PathJoinSubstitution([moveit_share, "config", "moveit.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        condition=IfCondition(launch_rviz),
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return [
        ur_control_launch,
        gripper_spawner,
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            choices=[
                "ur3", "ur5", "ur10", "ur3e", "ur5e", "ur7e", "ur10e",
                "ur12e", "ur16e", "ur8long", "ur15", "ur18", "ur20", "ur30",
            ],
        ),
        DeclareLaunchArgument("robot_ip", description="IP address of the UR robot"),
        DeclareLaunchArgument("use_mock_hardware", default_value="false"),
        DeclareLaunchArgument("headless_mode", default_value="false",
            description="Enable headless mode for robot control (skips waiting on the "
                        "polyscope 'play' button / external_control program)."),
        DeclareLaunchArgument("tf_prefix", default_value=""),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument(
            "description_package",
            default_value="arm_gripper_description",
            description="Package holding the COMBINED arm+Hand-E xacro (see file header note).",
        ),
        DeclareLaunchArgument(
            "description_file",
            default_value="arm_with_gripper.xacro",
            description="Combined arm+gripper xacro filename, under <description_package>/urdf/. "
                        "Must match the ACTUAL filename in your package — override with "
                        "description_file:=<name> if it's called something else.",
        ),
        DeclareLaunchArgument("moveit_config_package", default_value="ur_moveit_config"),
        DeclareLaunchArgument(
            "controllers_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("arm_gripper_description"), "config", "controllers.yaml"]
            ),
            description="Merged ros2_control controllers YAML — arm controllers + "
                        "gripper_action_controller in one file.",
        ),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
