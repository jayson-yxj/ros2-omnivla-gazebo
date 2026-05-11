from glob import glob
import os

from setuptools import setup

package_name = 'vlnav_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models', 'stable_diff_cam'), glob('models/stable_diff_cam/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh') + glob('scripts/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vln_project',
    maintainer_email='root@localhost',
    description='Gazebo MVP for VL-Nav-style language grounding and differential-drive navigation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_world_detector = vlnav_gazebo.yolo_world_detector:main',
            'scan_for_target = vlnav_gazebo.scan_for_target:main',
            'opentrackvla_policy = vlnav_gazebo.opentrackvla_policy:main',
            'omnivla_policy = vlnav_gazebo.omnivla_policy:main',
        ],
    },
)
