import os

image_name_dict = {
    'astronaut'       : 'astronaut.png',
    'bird'            : 'bird.jpg',
    'brick'           : 'brick.png',
    'camera'          : 'camera.png',
    'cat'             : 'cat.png',
    'chessboard_GRAY' : 'chessboard_GRAY.png',
    'chessboard_RGB'  : 'chessboard_RGB.png',
    'coffee'          : 'coffee.png',
    'coins'           : 'coins.png',
    'color'           : 'color.png',
    'dog'             : 'dog.png',
    'grass'           : 'grass.png',
    'gravel'          : 'gravel.png',
    'horse'           : 'horse.jpg',
    'motorcycle_left' : 'motorcycle_left.png',
    'motorcycle_right': 'motorcycle_right.png',
    'page'            : 'page.png',
}

def data_image_path(image_name=None):
    script_path = os.path.dirname(os.path.abspath(__file__))
    if image_name:
        image_name = image_name_dict[image_name]
        image_path = os.path.join(script_path, f'images/{image_name}')
    else:
        image_path = []
        all_image_paths = [image_name_dict[key] for key in image_name_dict]
        for img_name in all_image_paths:
            im_path = os.path.join(script_path, f'images/{img_name}')
            image_path.append(im_path)
    return image_path


if __name__=="__main__":
    from pyzjr.visualize import read_bgr, display
    print(data_image_path())
    img = read_bgr(data_image_path()[5])
    display(img)
