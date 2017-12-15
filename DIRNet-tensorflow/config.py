class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.use_saved_data = False,
    config.batch_size = 1
    config.im_size = [124, 124]
    config.lr = 1e-4
    config.iteration = 1000000

    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 10
    config.im_size = [124, 124]

    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
