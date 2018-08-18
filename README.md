# DenseNetFCN for segmentation of cloud images
______________________________________________

基于密集全连接网络的地基云图分割，本工程采用DenseNet作为骨干网络，在DenseNet提取的特征图之上采用FCN网络对云图实施分割。
结果如下图,中间图为标签右侧图为预测值：
<img src="https://github.com/huaifeng1993/Segmentation_of_Cloud_Images/blob/master/result/Figure_1.png" alt="1" align=center />
<img src="https://github.com/huaifeng1993/Segmentation_of_Cloud_Images/blob/master/result/Figure_2.png" alt="2" align=center />
<img src="https://github.com/huaifeng1993/Segmentation_of_Cloud_Images/blob/master/result/Figure_3.png" alt="3" align=center />

## 1.环境要求
    Python 3.4/3.5
    numpy
    scipy
    Pillow
    cython
    matplotlib
    scikit-image
    tensorflow>=1.3.0
    keras>=2.0.8
    opencv-python
    h5py
    imgaug
    IPython[all]
## 2文件结构
```
project 
  |--README.md  
  |--cloudData 
  |--DeFCN
      |--code   
         |--tmp
      |--result 
  ```
 ### 2.1文件目录说明
 * data存放训练数据。
 * code存放训练代码。测试代码。评价代码。
 * result存放测试图。
 * code/tmp 存放训练的模型
 ## 4.如何训练
 * 命令行运行de_main.py
 * 训练完成之后运行detection.py执行检测任务