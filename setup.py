import re
from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = [line for line in f.read().splitlines() 
                if line and re.match(r'^[^=]+==[^=]+$', line.strip())]

setup(
    name="khaosz", 
    version="1.2.0", 
    packages=find_packages(),
    install_requires=required,
    dependency_links=[
        "https://download.pytorch.org/whl/cu126",
    ],
    python_requires="==3.12.*",
)