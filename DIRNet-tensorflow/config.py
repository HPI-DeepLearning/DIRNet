class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.os_is_windows = False
  if is_train:
    config.use_saved_data = False
    config.batch_size = 1
    config.im_size = [222, 247]
    config.lr = 1e-4
    config.iteration = 170000  # ca 20 epochs
    config.s_dir="/home/adrian/Documents/dl2/Cardiac/ES_rescaled/"
    config.d_dir="/home/adrian/Documents/dl2/Cardiac/ED_rescaled/"
    config.label_path="./label.txt"
    config.s_data_filename="./s_data_save"
    config.d_data_filename="./d_data_save"
    config.label_filename="./label_save"
    config.s_data_eval_filename="./s_data_eval_save"
    config.d_data_eval_filename="./d_data_eval_save"
    config.label_eval_filename="./label_eval_save"
    config.save=True
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
    config.use_AffineST=False
    config.checkpoint_distance=50000
    config.result_dir = "result"
    config.eval_split_fraction = 0.2
  # else:
  #   config.use_saved_data = True
  #   config.s_dir="../Cardiac/ES"
  #   config.d_dir="../Cardiac/ED"
  #   config.s_data_filename="./s_data_save"
  #   config.d_data_filename="./d_data_save"
  #   config.save=True
  #   config.batch_size = 25
  #   config.im_size = [105, 128]
  #   config.use_AffineST=False
  #
  #   config.result_dir = "result"
  #   config.ckpt_dir = "ckpt"
  return config
