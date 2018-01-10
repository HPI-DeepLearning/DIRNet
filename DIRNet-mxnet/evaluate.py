import sys
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join
import scipy.ndimage as ndimage
import numpy as np
import helper as hlp
import scipy.misc as misc

def rmse(x, y):
    '''
    calculates the root mean squared error of two arrays
    :param x:
    :param y:
    :return:
    '''
    error = np.subtract(x, y)
    squared = np.square(error)
    avg = np.average(squared)
    rooted = np.sqrt(avg)
    return rooted


if __name__ == '__main__':
    # path_fixed = '/home/adrian/Documents/dl2/Cardiac/ED'
    path_es = '/home/adrian/Documents/dl2/Cardiac/ES'
    path_es_reg = '/home/adrian/Documents/dl2/Cardiac/ES_registered'
    path_ed_scaled = '/home/adrian/Documents/dl2/Cardiac/ES_rescaled'

    # path_to_root = './Registration/'
    # path_to_root = '/home/adrian/Documents/dl2/Cardiac/'
    # path_fixed = join(path_to_root, 'ED')
    # path_moving = join(path_to_root, 'ES')
    # out_fixed = join(path_to_root, 'ED_rescaled')
    # out_mov = join(path_to_root, 'ES_rescaled')
    # mkdir(out_fixed)
    # mkdir(out_mov)
    shape = (222, 247)
    sum = 0
    i = 0
    onlyfiles_fixed = [f for f in listdir(path_ed_scaled) if isfile(join(path_ed_scaled, f))]
    onlyfiles_moving = [f for f in listdir(path_es_reg) if isfile(join(path_es_reg, f))]
    for i, fixed in enumerate(onlyfiles_fixed):
        if fixed.endswith('.png'):
            moving = hlp.find_moving_img(onlyfiles_moving, i, fixed)
            assert moving is not None  # we have to search for the correct moving, cuz not same amnt of pics in ED, ES

            abspath = join(path_ed_scaled, fixed)
            pic_fix = ndimage.imread(abspath, flatten=True)
            abspath = join(path_es_reg, moving)
            pic_mov = ndimage.imread(abspath, flatten=True)
            sum += rmse(pic_fix, pic_mov)
    print('Average RMSE: ' + str(sum/i))