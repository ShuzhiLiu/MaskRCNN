import os

dir_list = ['SavedModels',
            'SavedDebugImages',
            'TensorboardLogs']

for directory in dir_list:
    if not os.path.isdir(directory):
        print(f"directory: {directory} doesn't exist, create now")
        os.mkdir(directory)
        print(f"created directory: {directory}")
