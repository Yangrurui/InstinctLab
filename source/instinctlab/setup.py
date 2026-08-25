"""Installation script for the 'instinctlab' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))
BACKEND_PINS = toml.load(os.path.join(EXTENSION_PATH, "config", "backend_pins.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
    "pytorch_kinematics",
    "joblib",
    "debugpy",
    "snakeviz",
    "trimesh[all]",
    "scikit-learn",
    "opencv-python",
    "packaging",
    "pyvista",
]

_ISAACLAB_GIT = "git+{git}@{commit}".format(**BACKEND_PINS["isaaclab"])
ISAACLAB_REQUIRES = [
    f"{name} @ {_ISAACLAB_GIT}#subdirectory=source/{name}" for name in BACKEND_PINS["isaaclab"]["packages"]
]
MJLAB_REQUIRES = [BACKEND_PINS["mjlab"]["pypi"], *BACKEND_PINS["mjlab"]["runtime"]]
EXTRAS_REQUIRE = {
    "isaaclab": ISAACLAB_REQUIRES,
    "mjlab": MJLAB_REQUIRES,
    "all": ISAACLAB_REQUIRES + MJLAB_REQUIRES,
}

# Installation operation
setup(
    name="instinctlab",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
