# Improved Pixel-Level Pavement-Defect Segmentation Using a Deep Autoencoder :page_facing_up: [[1]](#References) <br />
## :scroll: Article can be downloaded (open access) from: <br /> https://www.mdpi.com/1424-8220/20/9/2557 <br />
### If you find code or ideas useful, please cite [[1]](#References),[[2]](#References)

:fire: Information about training, prediction and computational performance can be found in the article :page_facing_up:.

:white_check_mark: Solutions:
  *  UNet
  *  ResUNet
  *  ResUNet with Atrous Spatial Pyramid Pooling
  *  ResUNet with Atrous Spatial Pyramid Pooling ("Waterfall"[[3]](#References) connection)
  *  ResUNet with Atrous Spatial Pyramid Pooling and Attention Gates
  *  ResUNet with Atrous Spatial Pyramid Pooling ("Waterfall"[[3]](#References) connection) and Attention Gates

:warning: It is not all! Feel free to make your own configuration using neural network block, defined in 'models/layers.py' :snake: file . You may find even more architectural solutions in the code than we mentioned :point_up: :eyes: :point_up:.

## Baseline model:  
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/UNet4.png" width="900"/>  

## Model induced with residual connections, ASPP, AG:  
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/UNet4AsppAGRes.png" width="900"/>  

## Few results with different architectures:
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_.png" width="425"/> <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_label_.png" width="425"/> <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_unet_.png" width="425"/>  <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_unetResWF_.png" width="425"/> 

## Third-party library priorities: 
 * Tensorflow (with Keras frontend)
 * OpenCV
 * Many nifty things related to Python programming language

## Usage:  
### Everything is straight-forward. Check comments in the code :eyes: 
 * train.py - train  
 * predict.py - predict  
 * predict_by_patches.py - predict a big image by cropping it into regions and joining them after  

# Rendered video results:  
---------------------  
## CrackForest links:  
#### [UNet](https://www.youtube.com/watch?v=mcLsCJ7fH2k&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=7), [ResUNet](https://www.youtube.com/watch?v=xEnShuqWLjg&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=11), [ResUNet+ASPP](https://www.youtube.com/watch?v=2sbeCc27ZUo&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=12), [ResUNet+ASPP+AG](https://www.youtube.com/watch?v=1AMHWY-OAhA&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=8), [ResUNet+ASPP_WF](https://www.youtube.com/watch?v=FqQiivLl1s8&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=10), [ResUNet+ASPP_WF+AG](https://www.youtube.com/watch?v=9F_zW5VmIT0&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=9)  
---------------------
## GAPs384 links:  
#### [UNet](https://www.youtube.com/watch?v=mjcN0ZoImzY&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=1), [ResUNet](https://www.youtube.com/watch?v=uTEA_Poum0E&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=6), [ResUNet+ASPP](https://www.youtube.com/watch?v=hfS3vNMW0Dc&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=5), [ResUNet+ASPP+AG](https://www.youtube.com/watch?v=6kOLpumZyHI&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=2), [ResUNet+ASPP_WF](https://www.youtube.com/watch?v=jlRIvxolr4Q&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=4), [ResUNet+ASPP_WF+AG](https://www.youtube.com/watch?v=a4f0V25L7qw&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=3)  
---------------------
## Crack500 links:  
#### [UNet](https://www.youtube.com/watch?v=k7cH-xb-_mA&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=13), [ResUNet](https://www.youtube.com/watch?v=M6uDWCY8l0Y&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=18), [ResUNet+ASPP](https://www.youtube.com/watch?v=vC6etYH93ug&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=17), [ResUNet+ASPP+AG](https://www.youtube.com/watch?v=kCyHJToBX-Q&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=14), [ResUNet+ASPP_WF](https://www.youtube.com/watch?v=V5jbJicdLzk&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=16), [ResUNet+ASPP_WF+AG](https://www.youtube.com/watch?v=D6D7cICRCF4&list=PL5dj7GxMk-6wtz5SVQnv1dPBoSc_5lHQ2&index=15)    
  
  --------------------
  ## References
  [1] Augustauskas, R.; Lipnickas, A. Improved Pixel-Level Pavement-Defect Segmentation Using a Deep Autoencoder. Sensors 2020, 20, 2557. <br />
  [2] Augustaukas, R.; Lipnickas, A. Pixel-wise Road Pavement Defects Detection Using U-Net Deep Neural Network. In Proceedings of the 2019 10th IEEE International Conference on Intelligent Data Acquisition and Advanced Computing Systems: Technology and Applications (IDAACS), Metz, France, 18–21 September 2019; IEEE: Metz, France, 2019; pp. 468–472 <br />
  [3] Artacho, B.; Savakis, A. Waterfall Atrous Spatial Pooling Architecture for Efficient Semantic Segmentation. Sensors 2019, 19, 5361. <br />
