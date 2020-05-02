# Improved Pixel-Level Pavement-Defect Segmentation Using a Deep Autoencoder [1]. <br />
## Article can be downloaded (open access) from https://www.mdpi.com/1424-8220/20/9/2557 <br />
## If you find code or ideas useful, please cite [1][2]

Solutions:
1. UNet
2. ResUNet
3. ResUNet with Atrous Spatial Pyramid Pooling
4. ResUNet with Atrous Spatial Pyramid Pooling ("Waterfall" connections)
5. ResUNet with Atrous Spatial Pyramid Pooling and Attention Gates
6. ResUNet with Atrous Spatial Pyramid Pooling ("Waterfall" connections) and Attention Gates

## Baseline model:  
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/UNet4.png" width="900"/>  

## Model induced with residual connections, ASPP, AG:  
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/UNet4AsppAGRes.png" width="900"/>  

## Some results with different architectures:
<img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_.png" width="425"/> <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_label_.png" width="425"/> <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_unet_.png" width="425"/>  <img src="https://github.com/rytisss/RoadPavementSegmentation/blob/master/res/20160222_164000_crack500_unetResWF_.png" width="425"/> 

Third-party library priorities: 
1. Tensorflow (with Keras frontend)
2. OpenCV
3. Many nifty things related to Python programming language

## Usage:  
### Everything is straight-forward. Check comments in the code!  
train.py - train  
predict.py - predict  
predict_by_patches.py - predict a big image by cropping it into regions and joining them after  

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
  [1] Augustauskas, R.; Lipnickas, A. Improved Pixel-Level Pavement-Defect Segmentation Using a Deep Autoencoder. Sensors 2020, 20, 2557.
  [2] Augustaukas, R.; Lipnickas, A. Pixel-wise Road Pavement Defects Detection Using U-Net Deep Neural Network. In Proceedings of the 2019 10th IEEE International Conference on Intelligent Data Acquisition and Advanced Computing Systems: Technology and Applications (IDAACS), Metz, France, 18–21 September 2019; IEEE: Metz, France, 2019; pp. 468–472
