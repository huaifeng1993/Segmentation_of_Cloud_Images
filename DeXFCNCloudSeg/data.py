
'''
data extraction for JD style recognition
aurhor:shenhuixiang
'''
from skimage import transform,io
import numpy as np
import logging
from Config import Config
import data_utils
import os
import random


"""参数:category决定数据类别为train validation test"""
data_path ='../cloudData'
test_path =''


"""参数:category决定数据类别为train validation test"""
class CloudDataSet(data_utils.Dataset):
    def load_Mete(self,dataset):
        try:
            assert dataset in ['training','val']
        except AssertionError as ae:
            print('the param \'dataset\' of the load_Mete fuction should be \'training\' or \'val\'.')

        GTmaps=os.listdir(os.path.join(data_path,'masks224'))
        GTmaps.sort()
        images=os.listdir(os.path.join(data_path,'images224'))
        images.sort()

        # 随机打乱图片顺序和标签顺序

        random.seed(32)
        random.shuffle(GTmaps)

        random.seed(32)
        random.shuffle(images)

        if(dataset=='training'):
            images=images[:19000]
            GTmaps=GTmaps[:19000]
        else:
            images = images[19000:]
            GTmaps = GTmaps[19000:]
        for img,mask in zip(images,GTmaps):
            #print(img_info)
            img_id=img
            img_path=os.path.join(data_path,'images224',img)
            mask_path=os.path.join(data_path,'masks224',mask)
            self.add_image("SEG",image_id=img_id,
                           path=img_path,mpath=mask_path,
                           width=224,height=224)

    def load_image(self,image_id):
        info=self.image_info[image_id]
        image=io.imread(info['path'])
        #image=transform.resize(image,(224,224))
        return image


    def load_label(self,image_id):
        info =self.image_info[image_id]
        img_label=info['img_label']
        return img_label

    def load_mask(self,image_id):
        info=self.image_info[image_id]
        img_mask=io.imread(info['mpath'])
        return img_mask/255
#################################################################
#batch 生成器
#################################################################


def load_image_mask(dataset,image_id,augment=False):
    """
    load image and label according to the image_id
    """
    image = dataset.load_image(image_id)
    mask = dataset.load_mask(image_id)
    return image,mask


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def data_generator(dataset,config,shuffle=True, augment=False):
    """A generator that returns images and corresponding target mask.

    dataset: The Dataset object to pick data from
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns one list, inputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - mask: [batch,H,W].
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.

            image_index = (image_index + 1) % len(image_ids)

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]
            #image_meta:image_id,image_shape,windows.active_class_ids
            image,img_mask=load_image_mask(dataset,image_id,augment)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (config.BATCH_SIZE ,)+ image.shape, dtype=np.float32)

                batch_masks = np.zeros((config.BATCH_SIZE,)+img_mask.shape,dtype=np.float32)

            batch_images[b] = image

            batch_masks[b] = img_mask
            b += 1

            # Batch full?
            # input_image,input_labels
            if b >= config.BATCH_SIZE:
                batch_masks=np.reshape(batch_masks,[config.BATCH_SIZE,224,224,1])
                inputs = (batch_images,batch_masks)
                yield inputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


if __name__=='__main__':
    config=Config()
    data_train=CloudDataSet()
    data_train.load_Mete('training')
    data_train.prepare()
    generator=data_generator(data_train,config=config)
    while True:
        next(generator)








