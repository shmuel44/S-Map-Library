from setuptools import setup
from setuptools import find_packages

long_description = '''
smaplib Python Library Package
'''

setup(
    name='smaplib',
    version='1.0.1',
    description='smaplib Python Library Package',
    long_description=long_description,
    author='Benjamin Raibaud',
    author_email='braibaud@gmail.com',
    url='https://github.com/braibaud/smaplib',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
        'annoy',
        'dataiku-internal-client',
        'pinecone-client',
        'tensorflow-hub',
        'tensorflow',
        'matplotlib',
        'pillow',
        'pandas',
    ],
    python_requires='>=3.6',
    packages=find_packages())