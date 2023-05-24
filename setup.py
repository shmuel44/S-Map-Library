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
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    # install_requires=[
    #     'annoy>=1.17.1,<2',
    #     'dataiku-internal-client>=11.0.0,<12',
    #     'pinecone-client>=2.2.1,<3',
    #     'tensorflow-hub>=0.12.0,<1',
    #     'tensorflow>=2.11.1,<3',
    #     'matplotlib>=3.7.1,<4',
    #     'pillow>=9.4.0,<10',
    #     'pandas>=1.5.3,<2'
    # ],
    python_requires='>=3.8',
    packages=find_packages())