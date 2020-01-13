class Param:
    # === can't change now ===
    N_STAGE = 5
    BATCH = 1
    # === could change now ===

    # --- RPN ---
    PATH_MODEL = 'SavedModels'
    LR = 0.001
    LAMBDA_FACTOR = 5
    EPOCH = 24
    IMG_SHAPE = (720,1280,3)
    ANCHOR_PROPOSAL_N = 300
    ANCHOR_THRESHOLD = 0.85
    RPN_NMS_THRESHOLD = 0.5
    # PATH_DATA = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
    # PATH_IMAGES = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images'
    PATH_DATA = '/home/liushuzhi/Documents/mmdetection_tools/data'
    PATH_IMAGES = '/home/liushuzhi/Documents/mmdetection_tools/LocalData_Images'
    # PATH_DATA = '/Users/shuzhiliu/Documents/mmdetection_tools/data'
    # PATH_IMAGES = '/Users/shuzhiliu/Documents/mmdetection_tools/LocalData_Images'
    DATASET_ID = '1940091026744'
