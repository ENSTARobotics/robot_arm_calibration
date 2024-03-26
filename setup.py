from setuptools import setup

setup(
   name='robot_arm_calibration',
   version='0.0.0',
   description='The robot_arm_calibration package',
   author='Caroline PASCAL',
   author_email='caroline.pascal.2020@ensta-paris.fr',
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
