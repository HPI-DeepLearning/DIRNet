from pathlib import Path
import os
import re
import shutil

path_src="./Medic_data/Alzeimer_small"
path_dest="./Medic_data/Alzeimer_ordered"



pathlist = Path(path_src).glob('**/*.png')
for image_path in pathlist:
    if dim_string not  in image_path:
        continue
    dest = str(image_path).split(".")[2]
    if not os.path.isdir(path_dest+"/"+dest):
        os.makedirs(path_dest+"/"+dest)
    shutil.copy(image_path, path_dest+"/"+dest)
