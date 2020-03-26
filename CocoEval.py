from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np



annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

#initialize COCO ground truth api
dataDir='Demo'
# dataType='val2014'
# annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
GroundTruthAnnotationFile = f"{dataDir}/annotations/test.json"
cocoGt=COCO(GroundTruthAnnotationFile)

#initialize COCO detections api
# resFile='%s/results/%s_%s_fake%s100_results.json'
ResultFile = f"{dataDir}/results/results.pkl.bbox.json"
# resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(ResultFile)

imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()