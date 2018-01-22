import numpy as np
import gzip
from pathlib import Path
# import imageio
from skimage.transform import resize
from skimage import color
import scipy.ndimage as ndimage
import scipy.misc
from config import get_config
import random
import h5py


class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train,config):
    self.is_train = is_train
    self.path = path
    self.s_data=[]
    self.d_data=[]
    self.config=config

    if not config.use_saved_data:
        self.s_data,s_data_names = self.get_data(self.config.s_dir)
        self.d_data,d_data_names = self.get_data(self.config.d_dir)
        # print('len of s_data: ' + str(s_data_names))
        # print('len of d_data: ' + str(d_data_names))
        index= 0
        to_be_deleted=[]
        for i in s_data_names:
            if i in d_data_names:
                index+=1
            else:
                to_be_deleted.append(index)
                index+=1
        self.s_data=np.delete(self.s_data, to_be_deleted,0)
        print('len of to_be_deleted: ' + str(len(to_be_deleted)))
        to_be_deleted=[]
        index=0
        for i in d_data_names:
            if i in s_data_names:
                index+=1
            else:
                to_be_deleted.append(index)
                index+=1
        self.d_data=np.delete(self.d_data, to_be_deleted,0)

        if self.config.save:
            with h5py.File('{}.h5'.format(self.config.s_data_filename), 'w') as hf:
                hf.create_dataset(self.config.s_data_filename,  data=self.s_data)
            with h5py.File('{}.h5'.format(self.config.d_data_filename), 'w') as hf:
                hf.create_dataset(self.config.d_data_filename,  data=self.d_data)
    else:
        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = '{}/{}'.format(prefix, key)
                if isinstance(item, h5py.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group): # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)
        with h5py.File(self.config.s_data_filename+'.h5', 'r') as hf:
             for (path, dset) in h5py_dataset_iterator(hf):
                 self.s_data=hf[dset.name][:]

        with h5py.File(self.config.d_data_filename+'.h5', 'r') as hf:
            for (path, dset) in h5py_dataset_iterator(hf):
                self.d_data=hf[dset.name][:]

  def extract_patientnumber(self,filepath):
      image_name=str(filepath).split("/")
      image_name=image_name[len(image_name)-1]
      num=image_name.split("_")[0][-3:]
      return num.lstrip("0")

  def get_data(self,path):
    pathlist = Path(path).glob('**/*.png')
    imagelist=[]
    pathnames=[]
    for image_path in pathlist:
        print(image_path)
        # maybe interesting at some  point
        slice_number=str(image_path).split(".")
        slice_number=slice_number[len(slice_number)-2]

        pathnames.append(self.extract_patientnumber(image_path)+slice_number)
        num = str(image_path).split(".")[2]
        #dropping that alpha channel...
        #res_im=resize(color.rgb2gray(imageio.imread(str(image_path))), self.config.im_size,mode='constant')
        res_im = ndimage.imread(str(image_path), flatten=True)
        imagelist.append(res_im)

        # png[int(num)].append(res_im)
    # png = [x for x in png if x != []]
    # for i in range(0,len(png)):
    #     png[i]=np.asarray(png[i])
    # #png=np.asarray(png)
    # # im = np.asarray(png)
    # # im = np.expand_dims(im, axis=3)
    # print ('Importing done...',len(png))
    imagelist=np.asarray(imagelist)
    imagelist=np.expand_dims(imagelist,axis=4)
    print(imagelist.shape)


    return imagelist, pathnames

  def sample_pair(self, batch_size, label=None):
    # print(len(self.s_data))
    # print(len(self.d_data))
    choice = np.random.choice(len(self.d_data)-1, batch_size)

    x = self.s_data[choice]
    y = self.d_data[choice]

    return x, y

  def get_pair_by_idx(self, idx):
    x = self.s_data[np.expand_dims(idx, 0)]
    y = self.d_data[np.expand_dims(idx, 0)]

    return x, y
