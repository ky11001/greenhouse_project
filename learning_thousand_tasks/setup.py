#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="thousand_tasks",
    version="1.0.0",
    author="Kamil Dreczkowski and Pietro Vitiello",
    author_email="krd115@ic.ac.uk and pv2017@ic.ac.uk",
    description="This repository contains the implementation of all methods evaluated in the paper \"Learning a Thousand Tasks in a Day\". We provide model architectures, training scripts, and deployment examples.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KamilDre/learning_thousand_tasks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "ros": [
            "rospy",
            "geometry_msgs",
            "std_msgs",
        ],
        "hardware": [
            "pyrealsense2",
            "robotiq-modbus-tcp",
        ]
    },
    entry_points={
        "console_scripts": [
            "thousand-tasks-train-act=thousand_tasks.training.act_end_to_end.train:main",
            "thousand-tasks-hybrid-bn=thousand_tasks.experiments.hybrid_bn:main",
            "thousand-tasks-mt3=thousand_tasks.experiments.mt3:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)