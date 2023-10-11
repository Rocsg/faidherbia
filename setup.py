from setuptools import setup, find_packages

setup(name='faidherbia',
      version='1.0.0',
      description='Tools for processing of aerial imaging for sustain sahel project in the context of Mansour thesis',
      url='https://github.com/mansourdiene10/faidherbia.git',
      author='Serigne Mansour Diene & Romain Fernandez',
      author_email='serignemansour.diene@univ-thies.sn',
      license='GPL2',
      install_requires=[
            'torch==2.0.1',
            'torchvision'
            'numpy',
            'matplotlib',
            'scikit-learn',
      ],
      packages=find_packages(),
      zip_safe=False)

