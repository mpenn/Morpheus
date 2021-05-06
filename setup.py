from setuptools import setup, find_packages

import versioneer

setup(
    name="morpheus",
    version="0.10",
    description="Morpheus",
    author="NVIDIA Corporation",
    packages=find_packages(include=["morpheus", "morpheus.*"]),
    license="Apache",
    cmdclass=versioneer.get_cmdclass()
)