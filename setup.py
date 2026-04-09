from setuptools import setup

#Following https://uoftcoders.github.io/studyGroup/lessons/python/packages/lesson/

#Look here https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
exec(open('lautools/version.py').read())

pkg_requires = ['denpy', 'numpy', 'zarr']

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='lautools',
    url='https://github.com/kulvait/KCT_lautools',
    author='Vojtech Kulvait',
    author_email='vojtech.kulvait@hereon.de',
    # Needed to actually package something
    packages=['lautools'],
    # Needed for dependencies
    install_requires=pkg_requires,
    entry_points={
    'console_scripts': [
        'removeHotPixels =  lautools.scripts.removeHotPixels:main',
    ]
    },
    # *strongly* suggested for sharing
    version=__version__,
    # The license can be anything you like
    license='GPL3',
    description='Python package for tomographic data preprocessing and analysis'
)
