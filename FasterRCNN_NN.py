from NN_Parts import Backbone, RPN
from Data_Helper import coco_tools
import numpy as np


class FasterRCNN():
    def __init__(self):
        b1 = Backbone()
        self.backbone_model = b1.backbone_model
        self.RPN = RPN(self.backbone_model)
        self.RPN_model = self.RPN.RPN_model




if __name__=='__main__':
    f1 = FasterRCNN()
    data1 = coco_tools(
        file='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data/1945415016934/annotations/train.json',
        imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
    img1 = data1.GetOriginalImage(image_id='20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474')
    t1, t2 = f1.RPN_model.predict(np.array([img1]))
    print(t1, t2)
        