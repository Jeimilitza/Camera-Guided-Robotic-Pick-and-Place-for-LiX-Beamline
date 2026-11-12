
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
     ur_type = LaunchConfiguration("ur_type")
     robot_ip = LaunchConfiguration("robot_ip")
     safety_limits = LaunchConfiguration("safety_limits")
     safety_pos_margin = LaunchConfiguration("safety_pos_margin")
     safety_k_position = LaunchConfiguration("safety_k_position")

     kinematics_params_file = LaunchConfiguration("kinematics_params_file")
     physical_params_file = LaunchConfiguration("physical_params_file")
     visual_params_file = LaunchConfiguration("visual_params_file")
     joint_limit_params_file = LaunchConfiguration("joint_limit_params_file")
     description_file = LaunchConfiguration("description_file")
     tf_prefix = LaunchConfiguration("tf_prefix")
     use_mock_hardware = LaunchConfiguration("use_mock_hardware")
     mock_sensor_commands = LaunchConfiguration("mock_sensor_commands")
     headless_mode = LaunchConfiguration("headless_mode")
     use_tool_communication = LaunchConfiguration("use_tool_communication")
     tool_parity = LaunchConfiguration("tool_parity")
     tool_baud_rate = LaunchConfiguration("tool_baud_rate")
     tool_stop_bits = LaunchConfiguration("tool_stop_bits")
     tool_rx_idle_chars = LaunchConfiguration("tool_rx_idle_chars")
     tool_tx_idle_chars = LaunchConfiguration("tool_tx_idle_chars")
     tool_device_name = LaunchConfiguration("tool_device_name")
     tool_tcp_port = LaunchConfiguration("tool_tcp_port")
     tool_voltage = LaunchConfiguration("tool_voltage")
     reverse_ip = LaunchConfiguration("reverse_ip")
     script_command_port = LaunchConfiguration("script_command_port")
     reverse_port = LaunchConfiguration("reverse_port")
     script_sender_port = LaunchConfiguration("script_sender_port")
     trajectory_port = LaunchConfiguration("trajectory_port")

    # Load description with necessary parameters
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("my_robot_cell_control"),
                    "urdf",
                    "my_robot_cell_controlled.urdf.xacro",
                ]
            ),
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "kinematics_parameters_file:=",
            kinematics_parameters_file,
            " ",
            "use_mock_hardware:=",
            use_mock_hardware,
            " ",
            "mock_sensor_commands:=",
            mock_sensor_commands,
            " ",
            "headless_mode:=",
            headless_mode,
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Typo/series of used UR robot.",
            choices=[
                "ur3",
                "ur3e",
                "ur5",
                "ur5e",
                "ur10",
                "ur10e",
                "ur16e",
                "ur20",
                "ur30",
            ],
            default_value="ur20",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip", description="IP address by which the robot can be reached."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "kinematics_parameters_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("my_robot_cell_control"),
                    "config",
                    "my_robot_calibration.yaml",
                ]
            ),
            description="The calibration configuration of the actual robot used.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Start robot with mock hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "mock_sensor_commands",
            default_value="false",
            description="Enable mock command interfaces for sensors used for simple simulations. "
            "Used only if 'use_mock_hardware' parameter is true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Enable headless mode for robot control",
        )
    )

    return LaunchDescription(
        declared_arguments
        + [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[robot_description],
            ),
        ]
    )
