import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    gui = LaunchConfiguration('gui')
    target_text = LaunchConfiguration('target_text')
    classes = LaunchConfiguration('classes')
    world = LaunchConfiguration('world')
    model_path = LaunchConfiguration('model_path')

    model_path_env = os.path.join(tb3_gazebo_share, 'models')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = model_path_env if not existing_model_path else f'{model_path_env}:{existing_model_path}'

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger_cam'),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),

        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('target_text', default_value='chair'),
        DeclareLaunchArgument('classes', default_value='chair,couch,sofa,table,bed,plant,door,television'),
        DeclareLaunchArgument('model_path', default_value='/root/Desktop/vln_project/models/yolov8s-worldv2.pt'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world'),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(gui),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_gazebo_share, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_gazebo_share, 'launch', 'spawn_turtlebot3.launch.py')),
            launch_arguments={'x_pose': '-3.0', 'y_pose': '1.0'}.items(),
        ),
        Node(
            package='vlnav_gazebo',
            executable='yolo_world_detector',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'image_topic': '/camera/image_raw',
                'model_path': model_path,
                'target_text': target_text,
                'classes': classes,
                'device': 'cuda:0',
                'max_hz': 1.0,
                'conf': 0.08,
            }],
        ),
        Node(
            package='vlnav_gazebo',
            executable='scan_for_target',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
