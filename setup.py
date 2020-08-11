#!/usr/bin/env python
from distutils.core import setup

setup(name="Poor Man's Stemmer",
      version='0.1',
      description='''Provides a python class for language stemming. 
      If you have a large vocabulary this module provides a good 
      (but not perfect) stemming tool.''',
      author='Rod Gomez',
      author_email='armyofthe penguin@gmail.com',
      url='https://github.com/rpgomez/poor_mans_stemmer',
      install_requires=['numpy','scipy'],
      py_modules=['pm_stemmer'])
