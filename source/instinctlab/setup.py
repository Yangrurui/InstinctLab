"""Installation script for the 'instinctlab' python package."""

import os

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(
    os.path.join(EXTENSION_PATH, "config", "extension.toml")
)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "instinctlab-engine-core==0.1.0",
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

ISAACLAB_REQUIRES = ["instinctlab-engine-isaacsim==0.1.0"]
MJLAB_REQUIRES = ["instinctlab-engine-mjlab==0.1.0"]
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
    entry_points={
        "instinctlab.assets": [
            "unitree_g1 = instinctlab.assets.unitree_g1.interface:native_module",
        ],
    },
    license=EXTENSION_TOML_DATA["package"]["license"],
    project_urls={
        "Documentation": "https://project-instinct.github.io/",
        "Source": EXTENSION_TOML_DATA["package"]["repository"],
        "Issues": f"{EXTENSION_TOML_DATA['package']['repository']}/issues",
    },
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
