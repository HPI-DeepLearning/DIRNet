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

use_saved_data=True

class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data_alzeimer=[]
    self.data_healthy=[]
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
          item = g[key]
          path = '{}/{}'.format(prefix, key)
          if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
          elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)
    if not use_saved_data:
        self.data_alzeimer = self._get_data("./Medic_data/Alzeimer","./Medic_data/alzeimer_save")
        self.data_healthy = self._get_data("./Medic_data/Healthy","./Medic_data/healthy_save")
    else:
        with h5py.File('./Medic_data/alzeimer_save.h5', 'r') as hf:
             for (path, dset) in h5py_dataset_iterator(hf):
                 self.data_alzeimer.append(hf[dset.name][:])

        with h5py.File('./Medic_data/healthy_save.h5', 'r') as hf:
            for (path, dset) in h5py_dataset_iterator(hf):
                self.data_healthy.append(hf[dset.name][:])
# def extract_images(filename):
#     """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         magic = _read32(bytestream)
#         if magic != 2051:
#             raise ValueError(
#                 'Invalid magic number %d in MNIST image file: %s' %
#                 (magic, filename))
#         num_images = _read32(bytestream)
#         rows = _read32(bytestream)
#         cols = _read32(bytestream)
#         buf = bytestream.read(rows * cols * num_images)
#         data = numpy.frombuffer(buf, dtype=numpy.uint8)
#         data = data.reshape(num_images, rows, cols, 1)
#         return data


  def _get_data(self,path,filename):
    pathlist = Path(path).glob('**/*.png')
    png = [list() for _ in range(0,300)]
    for image_path in pathlist:
        print(image_path)
        num = str(image_path).split(".")[2]
        #dropping that alpha channel...
        res_im=resize(imageio.imread(str(image_path)), [124,124],mode='constant')[:,:,:3]
        png[int(num)].append(res_im)
    png = [x for x in png if x != []]
    for i in range(0,len(png)):
        png[i]=np.asarray(png[i])
    #png=np.asarray(png)
    # im = np.asarray(png)
    # im = np.expand_dims(im, axis=3)
    print ('Importing done...',len(png))
    # with open(filename,'w') as f:
    #     pickle.dump(png,f)
    # print("saved list")

    # values, counts = np.unique(labels, return_counts=True)
    #
    # data = []
    # for i in range(10):
    #   label = values[i]
    #   count = counts[i]
    #   arr = np.empty([count, 28, 28, 1], dtype=np.float32)
    #   data.append(arr)
    #
    # l_iter = [0]*10
    # for i in range(labels.shape[0]):
    #   label = labels[i]
    #   data[label][l_iter[label]] = images[i] / 255.
    #   l_iter[label] += 1

    with h5py.File('{}.h5'.format(filename), 'w') as hf:
        i=0
        for k in png:
          hf.create_dataset(filename+str(i),  data=k)
          i=1+i

    return png

  def sample_pair(self, batch_size, label=None):
    rd = random.randint(0, min(len(self.data_alzeimer),len(self.data_healthy))-1)
    while len(self.data_alzeimer[rd])< rd or len(self.data_healthy[rd])< rd:
      rd = random.randint(0, min(len(self.data_alzeimer),len(self.data_healthy))-1)
    #print(rd)
    #batch_size=min(min(batch_size,len(self.data_alzeimer[rd])),len(self.data_healthy[rd]))
    choice1 = np.random.choice(self.data_alzeimer[rd].shape[0], batch_size)
    choice2 = np.random.choice(self.data_healthy[rd].shape[0], batch_size)
    x = self.data_alzeimer[rd][choice1]
    y = self.data_healthy[rd][choice2]

    return x, y
