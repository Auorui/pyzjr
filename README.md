# pyzjr

<h1 align="center">
<img src="https://github.com/Auorui/AI-Learning-Materials/blob/main/webbg/%E5%86%B0%E7%BA%A2%E8%8C%B6.png" width="300">
</h1><br>

[![Downloads](https://static.pepy.tech/badge/pyzjr)](https://pepy.tech/project/pyzjr)
[![Downloads](https://static.pepy.tech/badge/pyzjr/month)](https://pepy.tech/project/pyzjr)
[![Downloads](https://static.pepy.tech/badge/pyzjr/week)](https://pepy.tech/project/pyzjr)

This is a package that has been patched for excellent third-party libraries such as opencv and pytorch.

All useful functional functions written by individuals

I will also integrate algorithms I have previously written

## Installation

https://pypi.org/project/pyzjr/

https://github.com/Auorui/pyzjr (I want stars ⭐ hhh)

https://www.writebug.com/code/ae41e81c-adef-11ee-830a-0242ac140019/#

## Update log
`1.1.11` Significant changes have been made, including image augmentation, deep learning strategies, and the addition of 
         my own way of reading images.

`1.1.4` Added the dlearn section for deep learning. Currently, there are backbone networks, VOC dataset operations, 
        and some training strategies available.(There is a problem that has been resolved in pyzjr==1.1.5)

`1.1.2` Added basic open source feature extraction frameworks VGG, Resnet, Densenet.
        In addition, the advanced visual module only adds gesture recognition and requires 
        downloading the mediapipe. However, due to its complexity, there is no dependency added to pyzjr. 
        If you want to use it, you need to download it separately

`1.1.0` This update redefines the color module. Added some aspects of deep learning, such as vgg.

`1.0.0` Official upload of version 1.0.0

`0.0.19` Modified the problem of Escape character in the path read by the getPhotopath function

`0.0.17——0.0.18` Added Showimage module and added torch framework. It is recommended to install it in a virtual environment of deep learning framework.

`0.0.12——0.0.16` Modifying bugs and adding functions to the PIC.py file.

## Upload of Library

* `python setup.py sdist`

* `twine upload dist/*`