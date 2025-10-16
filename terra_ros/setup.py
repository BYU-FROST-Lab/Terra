from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'terra_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Chad Samuelson',
    maintainer_email='chadrs2@byu.edu',
    description='Package for running LIO-SAM and extracting LiDAR scans, synced RGB images, transformations and global point cloud maps',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'save_metric_data = terra_ros.save_metric_data:main',
        ]
    },
)
