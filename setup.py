from setuptools import setup, find_packages

import versioneer

setup(
    name="morpheus",
    version="0.10",
    description="Morpheus",
    classifiers=[
        "Development Status :: 3 - Alpha",

        # Utilizes NVIDIA GPUs
        "Environment :: GPU :: NVIDIA CUDA",

        # Audience (TODO: (MDD) Audit these)
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",

        # License
        "License :: OSI Approved :: Apache Software License",

        # Only support Python 3.8+
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    author="NVIDIA Corporation",
    packages=find_packages(include=["morpheus", "morpheus.*"]),
    include_package_data=True,
    install_requires=[
        "Click",
        "click-completion",
    ],
    license="Apache",
    python_requires='>=3.8, <4',
    cmdclass=versioneer.get_cmdclass(),
    entry_points='''
        [console_scripts]
        morpheus=morpheus.cli:run_cli
      ''',
)
