from setuptools import setup, find_packages

setup(
    name="corec_v4",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip() for line in open("requirements.txt")
    ],
    entry_points={
        "console_scripts": [
            "corec=main:main",
        ],
    },
)