from setuptools import setup, find_packages

setup(
    name='H-GASP',
    url='https://github.com/rebeccaceppas/H-GASP',
    author='Rebecca Ceppas de Castro',
    author_email='rebecca@ceppas.co',
    packages=find_packages(),
    install_requires=['h5py',
                      'numpy',
                      'yaml',
                      'os',  # maybe wont need
                      'numexpr',
                      'scipy',
                      'healpy',
                      'astropy']
)