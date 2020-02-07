import os

dir_list = ['SavedModels',
            'SavedDebugImages',
            'TensorboardLogs']

for directory in dir_list:
    if not os.path.isdir(directory):
        print(f"directory: {directory} doesn't exist, create now")
        os.mkdir(directory)
        print(f"created directory: {directory}")

command_list = []
command_list.append('pip install tensorflow matplotlib opencv-python')
command_list.append('pip install git+git://github.com/waspinator/pycococreator.git@0.2.0')
command_list.append('pip install cython')
command_list.append('pip install pycocotools')
for command in command_list:
    os.system(command=command)