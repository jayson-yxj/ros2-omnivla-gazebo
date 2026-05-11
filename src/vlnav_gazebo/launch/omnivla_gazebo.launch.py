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
    instruction = LaunchConfiguration('instruction')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    goal_yaw = LaunchConfiguration('goal_yaw')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    model_dir = LaunchConfiguration('model_dir')
    publish_cmd_vel = LaunchConfiguration('publish_cmd_vel')
    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    infer_hz = LaunchConfiguration('infer_hz')
    autostart_task = LaunchConfiguration('autostart_task')
    rviz = LaunchConfiguration('rviz')

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
    omnivla_model_dir = os.path.join(workspace_root, 'models', 'omnivla', 'omnivla-original')
    goal_image_path = os.path.join(omnivla_root, 'inference', 'goal_img.jpg')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = f'{package_models}:{small_house_models}'
    if existing_model_path:
        gazebo_model_path = f'{gazebo_model_path}:{existing_model_path}'

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),

        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value=small_house_world,
        ),
        DeclareLaunchArgument('instruction', default_value='move toward the goal'),
        DeclareLaunchArgument('goal_x', default_value='2.79818558693'),
        DeclareLaunchArgument('goal_y', default_value='-3.52509260178'),
        DeclareLaunchArgument('goal_yaw', default_value='-1.59079641132'),
        DeclareLaunchArgument('spawn_x', default_value='5.18359947205'),
        DeclareLaunchArgument('spawn_y', default_value='1.49744713306'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.32731997641'),
        DeclareLaunchArgument('model_dir', default_value=omnivla_model_dir),
        DeclareLaunchArgument('publish_cmd_vel', default_value='true'),
        DeclareLaunchArgument('max_linear', default_value='0.08'),
        DeclareLaunchArgument('max_angular', default_value='0.12'),
        DeclareLaunchArgument('infer_hz', default_value='1.0'),
        DeclareLaunchArgument('autostart_task', default_value='false'),
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
            executable='omnivla_policy',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'image_topic': '/camera/image_raw',
                'odom_topic': '/odom',
                'cmd_vel_topic': '/cmd_vel',
                'instruction': instruction,
                'goal_x': goal_x,
                'goal_y': goal_y,
                'goal_yaw': goal_yaw,
                'omnivla_root': omnivla_root,
                'model_dir': model_dir,
                'goal_image_path': goal_image_path,
                'publish_cmd_vel': publish_cmd_vel,
                'use_pose_goal': True,
                'use_language_goal': True,
                'use_image_goal': False,
                'infer_hz': infer_hz,
                'max_linear': max_linear,
                'max_angular': max_angular,
                'autostart_task': autostart_task,
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
