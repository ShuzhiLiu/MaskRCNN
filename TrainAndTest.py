from NN_Model import FasterRCNN

f1 = FasterRCNN()

# f1.train_RPN_RoI()
# f1.save_weight()
f1.load_weight()
f1.test_proposal_visualization()
f1.faster_rcnn_output()