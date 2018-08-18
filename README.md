# DenseNetFCN for segmentation of cloud images
______________________________________________

基于密集全连接网络的地基云图分割，本工程采用DenseNet作为骨干网络，在DenseNet提取的特征图之上采用FCN网络对云图实施分割。
结果如下图,中间图为标签右侧图为预测值：
！[1](/result/Figure_1.png)
！[2](/result/Figure_2.png)
！[3](/result/Figure_3.png)
！[4](/result/Figure_4.png)
！[5](/result/Figure_5.png)
！[6](/result/Figure_9.png)
！[7](/result/Figure_10.png)
！[8](/result/Figure_11.png)
！[9](/result/Figure_13.png)
！[10](/result/Figure_14.png)
！[1](/result/Figure_17.png)

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