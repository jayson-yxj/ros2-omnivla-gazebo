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
    world = LaunchConfiguration('world')
    instruction = LaunchConfiguration('instruction')
    open_trackvla_root = LaunchConfiguration('open_trackvla_root')
    hf_model_dir = LaunchConfiguration('hf_model_dir')
    qwen_model_name = LaunchConfiguration('qwen_model_name')
    dino_model_name = LaunchConfiguration('dino_model_name')
    siglip_model_name = LaunchConfiguration('siglip_model_name')
    device = LaunchConfiguration('device')
    auto_download = LaunchConfiguration('auto_download')
    publish_cmd_vel = LaunchConfiguration('publish_cmd_vel')

    model_path_env = os.path.join(tb3_gazebo_share, 'models')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = model_path_env if not existing_model_path else f'{model_path_env}:{existing_model_path}'

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger_cam'),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),

        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world'),
        ),
        DeclareLaunchArgument('instruction', default_value='follow the target person'),
        DeclareLaunchArgument('open_trackvla_root', default_value='/root/Desktop/vln_project/third_party/OpenTrackVLA'),
        DeclareLaunchArgument('hf_model_dir', default_value='/root/Desktop/vln_project/models/hf/omlab__opentrackvla-qwen06b'),
        DeclareLaunchArgument('qwen_model_name', default_value='Qwen/Qwen3-0.6B'),
        DeclareLaunchArgument('dino_model_name', default_value='facebook/dinov3-vits16-pretrain-lvd1689m'),
        DeclareLaunchArgument('siglip_model_name', default_value='google/siglip-so400m-patch14-384'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('auto_download', default_value='false'),
        DeclareLaunchArgument('publish_cmd_vel', default_value='true'),

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
            executable='opentrackvla_policy',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'image_topic': '/camera/image_raw',
                'cmd_vel_topic': '/cmd_vel',
                'instruction': instruction,
                'open_trackvla_root': open_trackvla_root,
                'hf_model_dir': hf_model_dir,
                'qwen_model_name': qwen_model_name,
                'dino_model_name': dino_model_name,
                'siglip_model_name': siglip_model_name,
                'device': device,
                'auto_download': auto_download,
                'publish_cmd_vel': publish_cmd_vel,
                'infer_hz': 0.5,
                'history': 31,
                'max_linear': 0.18,
                'max_angular': 0.55,
            }],
        ),
    ])
