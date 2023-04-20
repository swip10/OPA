import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(f_name):
    return open(os.path.join(os.path.dirname(__file__), f_name)).read()


setup(
    name='opa',
    version='0.1',
    description='Testing installation of Package',
    url='#',
    author='auth',
    author_email='author@email.com',
    license='MIT',
    packages=find_packages(),
    long_description=read('README.md'),
    zip_safe=False
)
