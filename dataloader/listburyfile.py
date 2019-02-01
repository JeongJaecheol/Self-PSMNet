import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    #filepath = "./data/Middleburt Datasets/"
    for (path, dir, files) in os.walk(filepath):
        if path.find('TRAIN') > -1 and path.find('rgb') > -1 and path.find('left') > -1:
            for filename in os.listdir(path + '/'):
                p = path + '/' + filename
                all_left_img.append(p)
                all_right_img.append(p.replace('left', 'right'))
                for filename_ in os.listdir(path.replace('rgb', 'disp') + '/'):
                    p_ = path.replace('rgb', 'disp') + '/' + filename_
                    all_left_disp.append(p_)

        elif path.find('TEST') > -1 and path.find('rgb') > -1 and path.find('left') > -1:
            for filename in os.listdir(path + '/'):
                p = path + '/' + filename
                test_left_img.append(p)
                test_right_img.append(p.replace('left', 'right'))
                for filename_ in os.listdir(path.replace('rgb', 'disp') + '/'):
                    p_ = path.replace('rgb', 'disp') + '/' + filename_
                    test_left_disp.append(p_)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp
