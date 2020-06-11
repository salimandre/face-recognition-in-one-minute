# Face Recognition In A Minute

We were inspired for this work by the following reference:

> *Online Semi-Supervised Perception: Real-Time Learning without Explicit Feedback*, Branislav Kveton, Michal Valko, Mathai Phillipose, Ling Huang, 2010


## Interest of the method

Our philosophy during this work was to be able to work in **real-time fashion** and to be **adaptative**. Therefore to avoid any heavy computation or training time. In some sense we wanted to produce a low-cost method of face recognition. In addition to the fact that only **light computation** is required it is **very little demanding on data**. We used a total of 60 imgs for graph of faces and a training set for PcaNet which consisted of less than 1000 imgs (we even could have used way less of them though). An other interest is the possibility to improve model by simply adding unlabelled imgs (hence **semi-supervised**) to graph. Also authors Branislav Kveton et al. claimed to obtain results using harmonic solution which are **better than using KNN** method.

Thus we used a pre-trained **Viola-Jones** algorithm *Robust Real-time Object Detection*, Paul Viola , Michael Jones, 2001 which has been thought to be fast and small device friendly. 

We also used **PcaNet** model which has a low amount of parameters (roughly 200) and is very fast to learn and yet produces good results (cf https://github.com/salimandre/PcaNet). Although we used our pre-trained PcaNet on LFW as of yet, it would be possible to perform training in real-time fashion.

To produce inference over new face imgs we need to compute one row of **similarity matrix** per unlabelled img then compute an **harmonic extension** of a function over the graph of faces. This takes basically O(|V|) for one frame.


## Our Pipeline

**inputs**: 
  * pre-trained Viola-Jones algorithm
  * a dataset of 1000 imgs from different people

**step 1**: choose 10 imgs from stars/personalities which are in some way (gender, age, skin color etc...) similar to you. We downloaded ours from google images.

These imgs will be added to Graph Of Faces and labelled as false.

<p align="center">
  <img src="img/my_stars.png" width="24%">
</p>

**step 2**: take few snapshots (5-10) in real-time of you, varying your pose and your distance to webcam 

These imgs will be added to Graph Of Faces and labelled as true.

**step 3**: compute bounding boxes using Viola-Jones algorithms to detect faces on every imgs.

Perform preprocessing step: crop, convert to grayscale, resize.

<p align="center">
 <img src="img/my_face.png" width="24%">
 <img src="img/my_face_pp.png" width="6%">
 <img src="img/my_stars_jesse.png" width="24%">
 <img src="img/jesse_pp.png.png" width="6%">
 <img src="img/my_stars_leo.png" width="24%">
 <img src="img/leo_pp.png.png" width="6%">
</p>

**step 4**: search of best PcaNet model for Graph Of Faces.
  
  - repeat a certain amount of times (5-10):
  
    - **Train PcaNet** on current true/false face images.
    - **Inference** of features on faces
    - **Evaluate** PcaNet model
     
      - repeat 100 times:
      
        - **sample** randomly an image from dataset:
      
        - **compute** one row of **similarity matrix** using L1 distance
        - <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=s(x,y)=(5000/d_{Manhattan}(x,y)))^4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(x,y)=(5000/d_{Manhattan}(x,y)))^4" title="s(x,y)=(5000/d_{Manhattan}(x,y)))^4" /></a></p >

        - <p align="center"><img src="img/simmat_1.png" width="60%"></p>
        - **solve** the following optimization problem:
        - <p align="center"><img src="img/eq_1.png" width="20%"></p>

        - by producing the harmonic solution:

        - <p align="center"><img src="img/eq_2.png" width="20%"></p>
        - compute the % of **false positive** 
   - Take **best model** with lowest false positive rate

**step 5**: Perform Boosting step by randomly choosing (10-50) faces from dataset which are barely classified as False by Graph Of Faces model. These nodes should be referred as boosting nodes or support nodes.

<p align="center">
  <img src="img/graph_22.png" width="60%">
</p>

**step 5**: Perform inference using boosting on frame.

For each frame perform "all in one" inference for both current frame and for faces chosen at previous steps.

## Results



## Limits

* Viola does not capture well changes of pose therefore to detect a face we need it to be mostly frontal

* we do not use a pre-trained segmentation tool to extract faces from background in bounding boxes.

* We used resised imgs (50,37) to feed into PcaNet in order to limit the number of features to 1920 (with our own current version of PcaNet we lack of flexibility) but with an other model (MobileNet?) it may be possible to use bigger imgs and have same quality of features or even better. 
