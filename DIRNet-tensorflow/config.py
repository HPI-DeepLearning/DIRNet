class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.use_saved_data = True
    config.batch_size = 50
    config.im_size = [105, 128]
    config.lr = 1e-4
    config.iteration = 1000000
    config.s_dir="../../Cardiac/ES"
    config.d_dir="../../Cardiac/ED"
    config.s_data_filename="./s_data_save"
    config.d_data_filename="./d_data_save"
    config.save=True
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
    config.use_AffineST=False
    config.checkpoint_distance=500
  else:
    config.use_saved_data = True
    config.s_dir="../Cardiac/ES"
    config.d_dir="../Cardiac/ED"
    config.s_data_filename="./s_data_save"
    config.d_data_filename="./d_data_save"
    config.save=True
    config.batch_size = 25
    config.im_size = [105, 128]
    config.use_AffineST=False

    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
