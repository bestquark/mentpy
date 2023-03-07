from setuptools import setup, find_packages

setup(
    name='mentpy',
    version='0.0.1-dev',
    author='Luis Mantilla',
    author_email='luismantilla99@outlook.com',
    description='A Python library for simulating small MBQC circuits',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BestQuark/mentpy',
    packages=find_packages(),
)
