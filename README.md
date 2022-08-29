# Msc-Project-2.5D-Image-Generation
GitHub Repo for the dissertation in part fulfilment of the Msc Degree in Data Science at University of Glasow. <br>
Two models have been trained to generate stereo pairs of images. The stereo pairs of images are used to get the depth map using stereo matching. Point cloud in painting is used to generate frames of image point cloud and are collated in the form of a video to show the parallax effect of a 2.5D image.<br>
## DCGAN
Deep Convolution Generative Adversarial Network (DCGAN) is trained on two datasets <br>
DrivingStereo and Holopix50k
The generator and Discriminator architecture is defined in 
<br> gennet.py and disnet.py <br>
To train the model, run train.py in the DCGAN directory. The models will be saved in the cpkt folder.
<br> For inference (Generate samples), run gen_img.py
<br> This will load the saved checkpoint and generate a sample from random noise.
## Projected FastGAN
FastGAN enerator is used with a Projected Discriminator
<br>The training script is in the projected_gan.ipynb notebook. The model is trained for 100000 iterations for a total time of 3 hrs 46 mins.
A snapshot of the network is saved to be used for inference later.
<br> run the gen_images.py script.specify the parameters
<br> --seeds="random seed-int", --nework="Path to the Netowrk", --outdir="Path to the output directory"
## Depth Estimation
The stereo pairs of images generated are passed to the depth estimation model
<br> run imageDepthEstimation.py and enter the path to the left and right images
![Left/right pair with depth map](https://github.com/Nerdy-Thanos/Msc-Project-2.5D-Image-Generation/blob/main/Screenshot%2020220829%20at%206.15.21%20PM.png)<br>
# 2.5D Parallax effect
run the autozoom.py after specifying the zoom parameters.
This generates a video file that shows the image with the depth parallax effect

# Acknowledgements
Parts of the code were adopted from the folllwing sources
- [DCGAN pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Projected GANs converge faster](https://github.com/autonomousvision/projected_gan)
- [Depth Estimation with stereo matching](https://github.com/ibaiGorordo/PyTorch-High-Res-Stereo-Depth-Estimation)
- [3d Ken Burns Effect](https://github.com/sniklaus/3d-ken-burns)

