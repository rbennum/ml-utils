from setuptools import setup, find_packages

setup(
    name="ml_utils",
    version="0.0.1",
    author="Bening Ranum",
    author_email="dev.rbennum@gmail.com",
    description="A collection of my personal utility functions for ML projects.",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "joblib"],
    python_requires=">=3.12",
)
