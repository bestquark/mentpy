from setuptools import setup, find_packages

version = "0.1.0a7"

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="mentpy",
    version=version,
    author="Luis Mantilla",
    author_email="luismantilla99@outlook.com",
    description="A Python library for simulating MBQC circuits",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BestQuark/mentpy",
    packages=find_packages(),
    install_requires=required_packages,
)
