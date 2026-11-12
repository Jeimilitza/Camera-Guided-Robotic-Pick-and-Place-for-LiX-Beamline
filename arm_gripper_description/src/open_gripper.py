#!/usr/bin/env python3

# ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py  set_tool_voltage:="24" use_fake_hardware:=false   create_socat_tty:=true   tty_port:=/tmp/ttyUR   socat_ip_address:=192.167.2.100   socat_port:=54321   frequency_hz:=10   launch_rviz:=false 

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import ParallelGripperCommand


ACTION_NAME = '/gripper_action_controller/gripper_cmd'


class GripperClient(Node):

    def __init__(self):
        super().__init__('gripper_client')

        self._client = ActionClient(
            self,
            ParallelGripperCommand,
            ACTION_NAME
        )

    def use_gripper(self, pos=0.025):
        self.get_logger().info(
            f'Waiting for action server: {ACTION_NAME}'
        )

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Action server not available.'
            )
            return False

        # Create the goal
        goal = ParallelGripperCommand.Goal()

        # The ParallelGripperCommand goal uses sensor_msgs/JointState.
        #
        # For the Robotiq Hand-E driver, the gripper position is
        # expressed through the command.position array.

        goal.command.position = [pos]

        # Send the goal
        self.get_logger().info(
            f"Sending command: position= {goal.command.position}!"
        )

        send_goal_future = self._client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_goal_future
        )

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error(
                'No goal handle returned.'
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                'Gripper goal was rejected.'
            )
            return False

        self.get_logger().info(
            'Gripper goal accepted.'
        )

        # Wait for the result
        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        result = result_future.result().result

        self.get_logger().info(
            f'Gripper finished: '
            f'reached_goal={result.reached_goal}, '
            f'stalled={result.stalled}'
        )

        return result.reached_goal


def main(args=None):

    rclpy.init(args=args)

    grip_node = GripperClient()

    time.sleep(1)
    grip_node.use_gripper(0.0085)
    time.sleep(2)
    
    grip_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py  set_tool_voltage:="24" use_fake_hardware:=false   create_socat_tty:=true   tty_port:=/tmp/ttyUR   socat_ip_address:=192.167.2.100   socat_port:=54321   frequency_hz:=10   launch_rviz:=false 

