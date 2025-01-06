from setuptools import setup

setup(
    name='decision-tree-algorithm',
    version='0.1',
    py_modules=['decision_tree'],
    install_requires=[
        'numpy', # List your dependencies here, like numpy
        'scikit-learn',  # if you're using scikit-learn
    ],
    entry_points={
        'console_scripts': [
            'decision-tree=decision_tree:main',
        ],
    },
)
