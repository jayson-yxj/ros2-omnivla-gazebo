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
    rviz = LaunchConfiguration('rviz')
    world = LaunchConfiguration('world')
    target_text = LaunchConfiguration('target_text')
    instruction = LaunchConfiguration('instruction')
    classes = LaunchConfiguration('classes')
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    omnivla_model_dir = LaunchConfiguration('omnivla_model_dir')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    detector_hz = LaunchConfiguration('detector_hz')
    detector_conf = LaunchConfiguration('detector_conf')
    target_min_score = LaunchConfiguration('target_min_score')
    target_min_area = LaunchConfiguration('target_min_area')
    target_only = LaunchConfiguration('target_only')
    infer_hz = LaunchConfiguration('infer_hz')
    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    visual_goal_timeout_sec = LaunchConfiguration('visual_goal_timeout_sec')
    visual_goal_stop_bottom_y = LaunchConfiguration('visual_goal_stop_bottom_y')
    autostart_task = LaunchConfiguration('autostart_task')

    package_share = get_package_share_directory('vlnav_gazebo')
    workspace_root = os.environ.get(
        'VLN_PROJECT_ROOT',
        os.path.dirname(os.path.dirname(os.path.dirname(package_share))),
    )
    robot_sdf = os.path.join(package_share, 'models', 'stable_diff_cam', 'model.sdf')
    rviz_config = os.path.join(package_share, 'config', 'omnivla_trajectory.rviz')
    package_models = os.path.join(package_share, 'models')
    omnivla_root = os.path.join(workspace_root, 'third_party', 'OmniVLA')
    small_house_root = os.path.join(workspace_root, 'third_party', 'aws-robomaker-small-house-world')
    small_house_models = os.path.join(small_house_root, 'models')
    small_house_world = os.path.join(small_house_root, 'worlds', 'small_house.world')
    default_omnivla_model = os.path.join(workspace_root, 'models', 'omnivla', 'omnivla-original')
    default_yolo_model = os.path.join(workspace_root, 'models', 'yolov8s-worldv2.pt')
    goal_image_path = os.path.join(omnivla_root, 'inference', 'goal_img.jpg')
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
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value=small_house_world),
        DeclareLaunchArgument('target_text', default_value='trash can,trash bin,garbage bin,waste bin,bin'),
        DeclareLaunchArgument('instruction', default_value='Move to the trash can and then stop'),
        DeclareLaunchArgument('classes', default_value=default_classes),
        DeclareLaunchArgument('yolo_model_path', default_value=default_yolo_model),
        DeclareLaunchArgument('omnivla_model_dir', default_value=default_omnivla_model),
        DeclareLaunchArgument('spawn_x', default_value='5.18359947205'),
        DeclareLaunchArgument('spawn_y', default_value='1.49744713306'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.32731997641'),
        DeclareLaunchArgument('detector_hz', default_value='4.0'),
        DeclareLaunchArgument('detector_conf', default_value='0.005'),
        DeclareLaunchArgument('target_min_score', default_value='0.55'),
        DeclareLaunchArgument('target_min_area', default_value='0.005'),
        DeclareLaunchArgument('target_only', default_value='true'),
        DeclareLaunchArgument('infer_hz', default_value='1.0'),
        DeclareLaunchArgument('max_linear', default_value='0.08'),
        DeclareLaunchArgument('max_angular', default_value='0.12'),
        DeclareLaunchArgument('visual_goal_timeout_sec', default_value='3.0'),
        DeclareLaunchArgument('visual_goal_stop_bottom_y', default_value='0.78'),
        DeclareLaunchArgument('autostart_task', default_value='false'),

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
                'model_path': yolo_model_path,
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
            executable='omnivla_policy',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'image_topic': '/camera/image_raw',
                'odom_topic': '/odom',
                'cmd_vel_topic': '/cmd_vel',
                'instruction': instruction,
                'goal_x': 0.0,
                'goal_y': 0.0,
                'goal_yaw': 0.0,
                'omnivla_root': omnivla_root,
                'model_dir': omnivla_model_dir,
                'goal_image_path': goal_image_path,
                'publish_cmd_vel': True,
                'use_pose_goal': False,
                'use_language_goal': True,
                'use_image_goal': False,
                'use_visual_goal_grounding': True,
                'visual_goal_timeout_sec': visual_goal_timeout_sec,
                'visual_goal_stop_bottom_y': visual_goal_stop_bottom_y,
                'infer_hz': infer_hz,
                'max_linear': max_linear,
                'max_angular': max_angular,
                'autostart_task': autostart_task,
                'stop_at_goal': True,
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
