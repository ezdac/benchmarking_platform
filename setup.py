from setuptools import setup, find_packages

setup(
    name='fipribench',
    version='0.1.0',
    # url='https://github.com/mypackage.git',
    # TODO add other authors
    author='Max Langenfeld, Sereina Riniker ',
    author_email='maxdacjs@gmail.com',
    description='Benchmarking platform for molecular fingerprints',
    packages=['fipribench'],
    install_requires=['numpy', 'matplotlib', 'sklearn'],  # TODO
    entry_points={
        'console_scripts': ['fipribench=fipribench.__main__:main'],
    }
)