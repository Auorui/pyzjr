"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.10
- pyzjr is the Codebase accumulated by my python programming.
  At present, it is only for my personal use. If you want to
  use it, please contact me. Here are my email and WeChat.
- WeChat: z15583909992
- Email: zjricetea@gmail.com
- Note: Currently still being updated, please refer to the latest version for any changes that may occur
"""
import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '1.1.2'
DESCRIPTION = ' A computer vision library that supports Win, packaged with patches for visual libraries such as \
                Opencv, matplotlib, and image. In the future, Pytorch will also be supported, all of which are personal (Auorui) \
                engineering code experience. '
LONG_DESCRIPTION = 'pyzjr is a computer vision library that supports Win'

setup(
    name="pyzjr",
    version=VERSION,
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
        'matplotlib',
        'scikit-image',
        'torch',
        'opencv-python',
        'tqdm',
        'torchvision',
        'torchsummary'
    ],
    keywords=['python', 'computer vision', 'pyzjr','windows'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ]
)
