import os
import sys
from pathlib import Path


def _project_root() -> str:
    env_root = os.environ.get('VLN_PROJECT_ROOT')
    if env_root:
        return env_root

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'install').is_dir() and (parent / 'src' / 'vlnav_gazebo').is_dir():
            return str(parent)
    return '/root/Desktop/vln_project'


PROJECT_ROOT = _project_root()

ROS_PYTHON_PATHS = [
    f'{PROJECT_ROOT}/install/lib/python3.10/site-packages',
    f'{PROJECT_ROOT}/install/local/lib/python3.10/dist-packages',
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
    '/opt/ros/humble/lib/python3.10/site-packages',
]

ROS_LIBRARY_PATHS = [
    f'{PROJECT_ROOT}/install/lib',
    '/opt/ros/humble/lib',
    '/opt/ros/humble/local/lib',
]


def ensure_ros_python():
    if sys.version_info[:2] == (3, 10) and _paths_available():
        return

    python = '/usr/bin/python3'
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = ':'.join(ROS_PYTHON_PATHS)
    library_paths = list(ROS_LIBRARY_PATHS)
    existing_library_path = env.get('LD_LIBRARY_PATH')
    if existing_library_path:
        library_paths.append(existing_library_path)
    env['LD_LIBRARY_PATH'] = ':'.join(library_paths)
    os.execve(python, [python, *sys.argv], env)


def _paths_available() -> bool:
    python_path = os.environ.get('PYTHONPATH', '')
    library_path = os.environ.get('LD_LIBRARY_PATH', '')
    return (
        any(path in python_path for path in ROS_PYTHON_PATHS)
        and any(path in library_path for path in ROS_LIBRARY_PATHS)
    )
