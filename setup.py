from os import path
from setuptools import find_packages, setup

# Get the absolute path of the current directory
this_directory = path.abspath(path.dirname(__file__))

# Read the contents of the README file
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# Remove lines containing images (e.g., PNG files) from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="DynaMimicGen",
    packages=[package for package in find_packages() if package.startswith("dmg")],
    install_requires=[
        "robosuite==1.4.1",
        "pyspacemouse",
        "hid",
        "numpy-quaternion==2023.0.4",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "cmeel-boost==1.83.0",
        "gymnasium==1.0.0",
        "pyyaml",
        "colorama",
        "requests",
        "cryptography",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco>=2.3.0",
        "spatialmath-python",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
        "jinja2",
        "typeguard",
        "h5py",
        "pytest-shutil",
        "tqdm",
        "einops",
        "cmake==3.31.6",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="DynaMimicGen: Synthetic dataset generation for manipulation tasks on the top of MuJoCo simulator.",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vincenzo Pomponi',
    author_email='vincenzo.pomponi@supsi.ch',
    python_requires='>=3.8',
)

