from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
setup_args = generate_distutils_setup(
    packages=['robot_arm_calibration'],
    package_dir={'': 'scripts'},
    install_requires=[
        #Arrays
        "numpy",
        #Rotation and optimization tools
        "scipy",
        #Plots
        "matplotlib",
        #Pickle file handling
        "dill",
        #Symbolic mathematics
        "sympy",
        #YAML file handling
        "pyyaml",
        #URDF handling and visualization
        #"urchin" #c.f. README
        "trimesh",
        #Combinatorial optimization
        "pygad",
        "simanneal",
        #Jacobian computation
        "jaxlib",
        "jax",
        #ROS packages folders recovery
        "rospkg",
        #YAML files
        "pyyaml",
    ]
)
setup(**setup_args)
