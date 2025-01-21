from pathlib import Path
from typing import Dict
from setuptools import find_packages, setup

VERSION: Dict[str, str] = {}

with open("datamaker/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

with Path("requirements.txt").open("tr") as reader:
    install_requires = [line.strip() for line in reader]

EXTRAS_REQUIRES: Dict[str, str] = {}
with Path("dev_requirements.txt").open("tr") as reader:
    EXTRAS_REQUIRES["dev"] = [line.strip() for line in reader]

setup(
    version=VERSION["VERSION"],
    name="symanto_datamaker",
    description="Package to generate synthetic data using GPT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Symanto Research GmbH",
    author_email="jose.gonzalez@symanto.com",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=EXTRAS_REQUIRES,
    python_requires=">=3.10.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    license_files=[
        "LICENSE",
    ],
)
