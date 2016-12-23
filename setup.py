from setuptools import setup, find_packages

setup(name='gym_recording',
      version='0.0.1',
      install_requires=['gym', 'boto3'],
      packages=find_packages(),
)
