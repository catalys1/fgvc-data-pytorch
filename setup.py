import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='fgvcdata',
    version='0.0.1',
    author='Connor Anderson',
    author_email='connor.anderson@byu.edu',
    description='PyTorch dataset classes with a common interface for '
                'Fine-Grained Visual Categorization datasets',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
    ),
)
