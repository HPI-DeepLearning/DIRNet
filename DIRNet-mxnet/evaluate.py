import sys
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join
import scipy.ndimage as ndimage
import numpy as np
import helper as hlp
import similarity as sim
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

def ncc(x,y):

    mean_x = np.mean(a=x, axis=(0, 1), keepdims=True)
    mean_y = np.mean(a=y, axis=(0, 1), keepdims=True)
    mean_x2 = np.mean(np.square(x), (0, 1), keepdims=True)
    mean_y2 = np.mean(np.square(y), (0, 1), keepdims=True)
    stddev_x = np.sum(np.sqrt(
        mean_x2 - np.square(mean_x)), (0, 1), keepdims=True)
    stddev_y = np.sum(np.sqrt(
        mean_y2 - np.square(mean_y)), (0, 1), keepdims=True)
    top = np.subtract(x, mean_x) * (np.subtract(y, mean_y))
    # return -np.mean(np.broadcast_div(top, np.broadcast_sub((stddev_x * stddev_y), 0.1)))
    return np.mean(np.divide((top), ((stddev_x * stddev_y))))

def difference(x,y):
    dif = np.subtract(x,y)
    abs = np.abs(dif)
    return abs

if __name__ == '__main__':
    # path_fixed = '/home/adrian/Documents/dl2/Cardiac/ED'
    path_es = '/home/adrian/Documents/dl2/Cardiac/ES'
    path_es_reg = '/home/adrian/Documents/dl2/Cardiac/ES_rescaled'
    path_ed_scaled = '/home/adrian/Documents/dl2/Cardiac/ED_rescaled'
    path_diff = '/home/adrian/Documents/dl2/Cardiac/ED_ES_reg_diff'

    # path_to_root = './Registration/'
    # path_to_root = '/home/adrian/Documents/dl2/Cardiac/'
    # path_fixed = join(path_to_root, 'ED')
    # path_moving = join(path_to_root, 'ES')
    # out_fixed = join(path_to_root, 'ED_rescaled')
    # out_mov = join(path_to_root, 'ES_rescaled')
    # mkdir(out_fixed)
    # mkdir(out_mov)
    shape = (222, 247)
    sum_rmse = 0
    sum_ncc = 0
    sum_ssim = 0
    i = 0
    onlyfiles_fixed = [f for f in listdir(path_ed_scaled) if isfile(join(path_ed_scaled, f))]
    onlyfiles_moving = [f for f in listdir(path_es_reg) if isfile(join(path_es_reg, f))]
    for fixed in onlyfiles_fixed:
        if fixed.endswith('.png'):
            moving = hlp.find_moving_img(onlyfiles_moving, i, fixed)
            assert moving is not None  # we have to search for the correct moving, cuz not same amnt of pics in ED, ES

            abspath = join(path_ed_scaled, fixed)
            pic_fix = ndimage.imread(abspath, flatten=True)
            abspath = join(path_es_reg, moving)
            pic_mov = ndimage.imread(abspath, flatten=True)
            i += 1
            sum_rmse += rmse(pic_fix, pic_mov)
            nvv = ncc(pic_fix, pic_mov)
            sum_ncc += nvv
            #misc.imsave(arr=difference(pic_fix, pic_mov), name=join(path_diff, fixed))
           # sum_ssim += sim.MultiScaleSSIM(pic_fix, pic_mov)
    print('Average RMSE: ' + str(sum_rmse / i))
    print('Average NCC: ' + str(sum_ncc / i))
    print('Average SSIM: ' + str(sum_ssim / i))