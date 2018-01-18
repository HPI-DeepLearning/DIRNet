import numpy as np
import gzip
from pathlib import Path
import imageio
from skimage.transform import resize
from skimage import color
import scipy.misc
from config import get_config
import random
import h5py


class DIRNetDatahandler(object):
    '''
    calculates the root mean squared error of two arrays
    :param config: config object
    '''

    def __init__(self, config):
        self.s_data = []
        self.d_data = []
        self.config = config

        # read data from folder
        if not config.use_saved_data:
            print("getting data")
            self.s_data, s_data_names = self.get_data(self.config.s_dir)
            self.d_data, d_data_names = self.get_data(self.config.d_dir)
            index = 0

            # delete files that are only contained in one of the folders
            to_be_deleted = []
            for i in s_data_names:
                if i in d_data_names:
                    index += 1
                else:
                    to_be_deleted.append(index)
                    index += 1
            self.s_data = np.delete(self.s_data, to_be_deleted, 0)

            # delete files that are only contained in one of the folders
            to_be_deleted = []
            index = 0
            for i in d_data_names:
                if i in s_data_names:
                    index += 1
                else:
                    to_be_deleted.append(index)
                    index += 1
            self.d_data = np.delete(self.d_data, to_be_deleted, 0)

            # save numpy arrays as .h5 if config.save is true
            if self.config.save:
                with h5py.File('{}.h5'.format(self.config.s_data_filename), 'w') as hf:
                    hf.create_dataset(self.config.s_data_filename, data=self.s_data)
                with h5py.File('{}.h5'.format(self.config.d_data_filename), 'w') as hf:
                    hf.create_dataset(self.config.d_data_filename, data=self.d_data)
        else:

            # load numpy arrays from .h5 files
            def h5py_dataset_iterator(g, prefix=''):
                for key in g.keys():
                    item = g[key]
                    path = '{}/{}'.format(prefix, key)
                    if isinstance(item, h5py.Dataset):  # test for dataset
                        yield (path, item)
                    elif isinstance(item, h5py.Group):  # test for group (go down)
                        yield from h5py_dataset_iterator(item, path)

            # load s_data
            with h5py.File(self.config.s_data_filename + '.h5', 'r') as hf:
                for (path, dset) in h5py_dataset_iterator(hf):
                    self.s_data = hf[dset.name][:]

            # load d_data
            with h5py.File(self.config.d_data_filename + '.h5', 'r') as hf:
                for (path, dset) in h5py_dataset_iterator(hf):
                    self.d_data = hf[dset.name][:]

    def extract_patientnumber(self, filepath):
        '''
        extract patient number from  filename
        :param filepath: path to file
        return: patientnumber
        '''

        image_name = str(filepath).split("\\")
        image_name = image_name[len(image_name) - 1]
        num = image_name.split("_")[0][-3:]
        return num.lstrip("0")

    def get_data(self, path):
        '''
        load images from path into numpy array
        :param path: path to folder
        return: numpy array with images and list with pathnames (images,pathnames)
        '''

        pathlist = Path(path).glob('**/*.png')
        imagelist = []
        pathnames = []
        for image_path in pathlist:
            print(image_path)
            # maybe interesting at some  point
            slice_number = str(image_path).split(".")
            slice_number = slice_number[len(slice_number) - 2]

            pathnames.append(self.extract_patientnumber(image_path) + slice_number)
            num = str(image_path).split(".")[2]

            # load images from file; rgb-> grayscale; resize to size defined in config.im_size
            res_im = resize(color.rgb2gray(imageio.imread(str(image_path))), self.config.im_size, mode='constant')
            imagelist.append(res_im)

        # list to numpy array
        imagelist = np.asarray(imagelist)

        # create colorchannel for grayscaled images
        # not needed if rgb is used -> comment this  line out
        imagelist = np.expand_dims(imagelist, axis=4)

        return imagelist, pathnames

    def sample_pair(self, batch_size):
        '''
        sample random pairs of moving and fixed images
        :param batch_size: number of moving/fixed images to be fixed
        return: numpy arrays x and y with shape [batch_size, height, width,color_channels]
        '''
        choice = np.random.choice(len(self.d_data) - 1, batch_size)

        x = self.s_data[choice]
        y = self.d_data[choice]

        return x, y
