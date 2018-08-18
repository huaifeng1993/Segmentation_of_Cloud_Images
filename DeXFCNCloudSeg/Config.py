import numpy as np

###########################################################
#Config
##########################################################
class Config(object):
    """DenseNet=
    """
    NAME = 'CLOUD'
    BATCH_SIZE = 8
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    DenseNet = [6, 12, 24, 16] # the params "[6, 12, 24, 16],[6, 12, 32, 32],[6, 12, 48, 32]" maps to DensNet "121,169,201"
    INPUT_SHAPE=[224,224,3]
    # Image mean (RGB)
    MEAN_PIXEL = np.array([170.8, 162.6, 159.5])

    # Learning rate and momentum
    # The DeNet paprer uses lr=0.1 .The lr is divided by 10 at
    # 50% and 75 of the number of training epochs.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

if __name__=='__main__':
    config=Config()
    print(config.STEPS_PER_EPOCH)