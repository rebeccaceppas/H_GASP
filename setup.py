from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = ''

setup(
    name='H-GASP',
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/rebeccaceppas/H-GASP',
    author='Rebecca Ceppas de Castro',
    author_email='rebecca@ceppas.co',
    packages=find_packages(),
    install_requires=['h5py',
                      'numpy',
                      'pyyaml',
                      'numexpr',
                      'scipy',
                      'healpy',
                      'astropy'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)