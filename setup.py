from setuptools import setup, find_packages

import re
import os

version = '0.0.0'

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

try:
    import pypandoc
    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='stacked_generalization',
      version=version,
      description='Machine Learning Stacking Uti',
      keywords = 'Stacking, Machine Learning',
      author='Ryosuke Fukatani',
      author_email='nannyakannya@gmail.com',
      url='https://github.com/fukatani/stacked_generalization',
      license="Apache License 2.0",
      packages=find_packages(),
      package_data={ 'stacked_generalization' : ['testcode/*'], },
      long_description=read_md('Readme.md'),
)

