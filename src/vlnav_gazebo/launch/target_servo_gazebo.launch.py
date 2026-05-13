import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')
    target_text = LaunchConfiguration('target_text')
    classes = LaunchConfiguration('classes')
    model_path = LaunchConfiguration('model_path')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    detector_hz = LaunchConfiguration('detector_hz')
    detector_conf = LaunchConfiguration('detector_conf')
    target_min_score = LaunchConfiguration('target_min_score')
    target_min_area = LaunchConfiguration('target_min_area')
    target_only = LaunchConfiguration('target_only')
    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    scan_angular = LaunchConfiguration('scan_angular')
    target_area_stop = LaunchConfiguration('target_area_stop')
    autostart = LaunchConfiguration('autostart')
    rviz = LaunchConfiguration('rviz')

    package_share = get_package_share_directory('vlnav_gazebo')
    workspace_root = os.environ.get(
        'VLN_PROJECT_ROOT',
        os.path.dirname(os.path.dirname(os.path.dirname(package_share))),
    )
    robot_sdf = os.path.join(package_share, 'models', 'stable_diff_cam', 'model.sdf')
    rviz_config = os.path.join(package_share, 'config', 'target_servo.rviz')
    package_models = os.path.join(package_share, 'models')
    small_house_root = os.path.join(workspace_root, 'third_party', 'aws-robomaker-small-house-world')
    small_house_models = os.path.join(small_house_root, 'models')
    small_house_world = os.path.join(small_house_root, 'worlds', 'small_house.world')
    yolo_model = os.path.join(workspace_root, 'models', 'yolov8s-worldv2.pt')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = f'{package_models}:{small_house_models}'
    if existing_model_path:
        gazebo_model_path = f'{gazebo_model_path}:{existing_model_path}'

    default_classes = (
        'trash can,trash bin,garbage bin,waste bin,bin,'
        'chair,couch,sofa,table,bed,plant,door,television'
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),

        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('world', default_value=small_house_world),
        DeclareLaunchArgument('target_text', default_value='trash can,trash bin,garbage bin,waste bin,bin'),
        DeclareLaunchArgument('classes', default_value=default_classes),
        DeclareLaunchArgument('model_path', default_value=yolo_model),
        DeclareLaunchArgument('spawn_x', default_value='5.18359947205'),
        DeclareLaunchArgument('spawn_y', default_value='1.49744713306'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.32731997641'),
        DeclareLaunchArgument('detector_hz', default_value='4.0'),
        DeclareLaunchArgument('detector_conf', default_value='0.005'),
        DeclareLaunchArgument('target_min_score', default_value='0.55'),
        DeclareLaunchArgument('target_min_area', default_value='0.005'),
        DeclareLaunchArgument('target_only', default_value='true'),
        DeclareLaunchArgument('max_linear', default_value='0.05'),
        DeclareLaunchArgument('max_angular', default_value='0.14'),
        DeclareLaunchArgument('scan_angular', default_value='0.08'),
        DeclareLaunchArgument('target_area_stop', default_value='0.11'),
        DeclareLaunchArgument('autostart', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(gui),
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'stable_diff_cam',
                '-file', robot_sdf,
                '-x', spawn_x,
                '-y', spawn_y,
                '-z', '0.01',
                '-Y', spawn_yaw,
            ],
            output='screen',
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
                'max_hz': detector_hz,
                'conf': detector_conf,
                'target_min_score': target_min_score,
                'target_min_area': target_min_area,
                'target_only': target_only,
            }],
        ),
        Node(
            package='vlnav_gazebo',
            executable='target_servo',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'target_text': target_text,
                'max_linear': max_linear,
                'max_angular': max_angular,
                'scan_angular': scan_angular,
                'target_area_stop': target_area_stop,
                'autostart': autostart,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(rviz),
        ),
    ])
