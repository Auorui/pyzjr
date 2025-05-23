"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.10
- pyzjr is the Codebase accumulated by my python programming.
  At present, it is only for my personal use. If you want to
  use it, please contact me. Here are my email and WeChat.
- WeChat: zjricetea
- Email: zjricetea@gmail.com
- Note: Currently still being updated, please refer to the latest version for any changes that may occur

    python setup.py sdist
    twine upload dist/*

    git init
    git branch -m master main
    git add .
    git commit -m "first commit"
    git remote add origin git@github.com:Auorui/pyzjr.git
    git push -u origin main
    # git push origin main --force
    # git fetch origin main
    # git pull origin main

"""
import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you don't need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

DESCRIPTION = ' A computer vision library that supports Win, packaged with patches for visual libraries such as \
                Opencv, matplotlib, and image. In the future, Pytorch will also be supported, all of which are personal (Auorui) \
                engineering code experience. '
LONG_DESCRIPTION = 'pyzjr is a computer vision library that supports Win'

setup(
    name="pyzjr",
    version='1.4.14',
    author="Auorui",
    author_email='zjricetea@gmail.com',
    url='https://github.com/Auorui/pyzjr',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'requests',
        'tqdm',
        'seaborn',
        'opencv-python',
        'scikit-learn',
        'scikit-image',
        'torch',
        'thop',
        'argparse',
        'pyyaml',
        'tensorboard',
        'psutil',
        'pytorch_wavelets',
        'torchvision',
    ],
    keywords=['python', 'computer vision', 'pyzjr', 'windows'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ]
)
