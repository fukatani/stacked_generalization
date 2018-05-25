import io
import os
from setuptools import setup, find_packages

version = '0.0.6'

install_requires = [
    'numpy',
    'scikit-learn',
    'pandas',
]

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    return io.open(os.path.join(CURRENT_DIR, filename), encoding='utf-8').read()

setup(name='stacked_generalization',
      version=version,
      description='Machine Learning Stacking Util',
      keywords = 'Stacking, Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/stacked_generalization',
      license="Apache License 2.0",
      packages=find_packages(),
      package_data={ 'stacked_generalization' : ['Readme.md'], },
      long_description='Readme.rst',
      install_requires=install_requires,
)

