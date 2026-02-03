from setuptools import setup, find_packages

setup(
    name="Topsis-Gaurang-102303907",
    version="1.0.0",
    author="Gaurang",
    author_email="your-gmangla_be23@gmail.com",
    description="A Python package for TOPSIS implementation",
    packages=find_packages(),
    install_requires=["pandas", "numpy"],
    entry_points={
        "console_scripts": [
            "topsis=topsis.topsis:main"
        ]
    },
)