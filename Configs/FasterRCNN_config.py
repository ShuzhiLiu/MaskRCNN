import numpy as np
class Param:
    # === can't change now ===
    N_STAGE = 5
    BATCH_RPN = 1
    BATCH_RoI = 4
    # === could change now ===

    # --- Anchor generator ---
    BASE_SIZE = 12
    RATIOS = None
    SCALES = 2 ** np.arange(3, 6)
    N_ANCHORS = 9

    # --- NN train data generator ---
    THRESHOLD_IOU_RPN = 0.7
    THRESHOLD_IOU_RoI = 0.55

    # --- RPN ---
    PATH_MODEL = 'SavedModels'
    PATH_DEBUG_IMG = 'SavedDebugImages'
    LR = 0.0001         # Currently, it doesn't work when lr > 0.0001
    LAMBDA_FACTOR = 1   # Don't change now! This factor is for balancing the RPN losses. 1 is the best now!!
    EPOCH = 12
    IMG_ORIGINAL_SHAPE = (720, 1280, 3)
    IMG_RESIZED_SHAPE = (800, 1333, 3)

    ANCHOR_PROPOSAL_N = 300
    ANCHOR_THRESHOLD = 0.51
    RPN_NMS_THRESHOLD = 0.4

    # --- RoI ---
    N_OUT_CLASS = 80


    DATA_JSON_FILE = '/media/liushuzhi/HDD500/Dataset/COCO2017/annotations/instances_val2017_sample.json'
    PATH_IMAGES = '/media/liushuzhi/HDD500/Dataset/COCO2017/val2017'

