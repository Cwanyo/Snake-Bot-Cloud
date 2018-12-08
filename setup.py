from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Snake-Bot',
      author='Chatchawan Yoojuie',
      author_email='manwan444@gmail.com',
      license='MIT',
      install_requires=[
          'keras', 'h5py', 'pygame', 'numpy', 'matplotlib', 'tensorflow'
      ],
      zip_safe=False)
