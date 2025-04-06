from setuptools import setup, find_packages

setup(
    name='TritonHub',
    version='1.0.4',
    author='Ali Youssef',
    description='A collection of Triton-based neural network modules',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ayoussf/Triton-Hub',
    packages=find_packages(exclude=['build', 'dist', 'TritonHub.egg-info', 'UnitTests']),
    python_requires='>=3.7',
    install_requires=['torch',
                      'triton'],
    )