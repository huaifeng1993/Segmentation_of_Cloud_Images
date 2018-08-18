from densenet import *
import matplotlib.pyplot as plt

from Config import Config
from data import CloudDataSet



from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#Directory to save logs and trained model
MODEL_DIR='./tmp'


#Configuration for training on the Cloud dataset.
config=Config()

#Validation dataset
data_val = CloudDataSet()
data_val.load_Mete('val')
data_val.prepare()

model=DeFCN(mode="inference",config=config,model_dir=MODEL_DIR)
model.load_weights(model.find_last(),by_name=True)
aTP=0
aFP=0
aTN=0
aFN=0
for id in data_val.image_ids:
    image = data_val.load_image(id)
    gt_map = data_val.load_mask(id)

    pre=model.detect(image)
    pre.astype(np.int)
    pre=np.reshape(pre,(224,224))

    FP=pre-gt_map

    FP[np.where(FP==-1)]=0
    TP=pre-FP
    FN=gt_map-TP
    TN=1-FN-FP-TP

    aTP=np.sum(TP)+aTP
    aFP=np.sum(FP)+aFP
    aTN=np.sum(TN)+aTN
    aFN=np.sum(FN)+aFN
    # P=np.sum(TP)/(np.sum(TP)+np.sum(FP))
    # R=np.sum(TP)/(np.sum(TP)+np.sum(FN))
    # F1=2*P*R/(P+R)
    print(id)
P=np.sum(aTP)/(np.sum(aTP)+np.sum(aFP))
R=np.sum(aTP)/(np.sum(aTP)+np.sum(aFN))
F1=2*P*R/(P+R)
MRate=(aFP+aFN)/(aTP+aTN+aFP+aFN)
print(P,R,F1,MRate)
#loading image for  test.
# image=data_val.load_image(1150)
# gt_map=data_val.load_mask(1150)
#feed the image to model
# pre=model.detect(image)
# pre.astype(np.int)
# pre=np.reshape(pre,(224,224))
#
# plt.subplot(131)
# plt.imshow(image)
# plt.subplot(132)
# plt.imshow(gt_map)
# plt.subplot(133)
# plt.imshow(pre)
# plt.show()
