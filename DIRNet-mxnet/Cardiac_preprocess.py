import sys
from os import listdir, mkdir
from os.path import isfile, join
import scipy.ndimage as ndimage
import numpy as np
import scipy.misc as misc


def find_moving_img(arr, start_idx, fixed_name):
    # patient035_frame01.nz.10.png
    patient_id = fixed_name[7:10]
    slice_id = fixed_name[22:24]
    for i in range(len(arr)):
        idx = (i + start_idx) % len(arr)  # iterate through the whole array but dont start at 0
        moving_name = arr[idx]
        if patient_id == moving_name[7:10] and slice_id == moving_name[22:24]:
            return moving_name
    return None


if __name__=='__main__':
    '''
    Reads the fixed (ED) cardio images, looks for a corresponding moving image (ES) by name (same patient, same slice)
    rescales and writes as greyscale to disk
    '''
    path_to_root = './Registration/'
    path_to_root = '/home/adrian/Documents/dl2/Cardiac/'
    path_fixed = join(path_to_root, 'ED')
    path_moving = join(path_to_root, 'ES')
    out_fixed = join(path_to_root, 'ED_rescaled')
    out_mov = join(path_to_root, 'ES_rescaled')
    mkdir(out_fixed)
    mkdir(out_mov)
    shape = (222, 247)
    i=0

    onlyfiles_fixed = [f for f in listdir(path_fixed) if isfile(join(path_fixed, f))]
    onlyfiles_moving = [f for f in listdir(path_moving) if isfile(join(path_moving, f))]
    for i, fixed in enumerate(onlyfiles_fixed):
        if fixed.endswith('.png'):
            moving = find_moving_img(onlyfiles_moving, i, fixed)
            assert moving is not None #we have to search for the correct moving, cuz not same amnt of pics in ED, ES

            abspath = join(path_fixed, fixed)
            pic_fix = ndimage.imread(abspath, flatten=True)
            pic_fix = misc.imresize(pic_fix, (shape[0], shape[1]))
            misc.imsave(arr=pic_fix, name=join(out_fixed, fixed))

            abspath = join(path_moving, moving)
            pic_mov = ndimage.imread(abspath, flatten=True)
            pic_mov = misc.imresize(pic_mov, (shape[0], shape[1]))
            misc.imsave(arr=pic_mov, name=join(out_mov, moving))
    print(str(i) + 'pics rescaled')