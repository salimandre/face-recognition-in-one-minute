# Face Recognition In A Minute

We were inspired for this work by the following reference:

> *Online Semi-Supervised Perception: Real-Time Learning without Explicit Feedback*, Branislav Kveton, Michal Valko, Mathai Phillipose, Ling Huang, 2010


## Interest of the method

Our philosophy during this work was to be able to work in **real-time fashion** and to be **adaptative**. Therefore to avoid any heavy computation or training time. In some sense we wanted to produce a low-cost method of face recognition. In addition to the fact that only **light computation** is required it is **very little demanding on data**. We used a total of 1200 imgs but we could make it work with way less. An other intereset is the possibility to improve model by simply adding unlabelled imgs (hence **semi-supervised**) to graph. Also authors Branislav Kveton et al. claimed to obtain results using harmonic solution which are **better than using KNN** method.

Thus we used a pre-trained Viola-Jones algorithm *Robust Real-time Object Detection*, Paul Viola , Michael Jones, 2001 which has been thought to be fast and small device friendly. 

We also used PcaNet model which has a low amount of parameters (roughly 200) and is very fast to learn and yet produces good results (cf https://github.com/salimandre/PcaNet). Although we used our pre-trained PcaNet on LFW as of yet, it would be possible to perform training in real-time fashion.

To produce inference over new face imgs we need to compute one row of similarity matrix per unlabelled img then compute an harmonic extension of a function over the graph of faces. This takes basically O(|V|) for one frame.


## Our Pipeline

inputs: 
  * pre-trained PcaNet over 1000 imgs (LFW)
  * Manhattan distance (L1 distance)
  * pre-trained Viola-Jones algorithm
  * 50 imgs from different people (label = false)

step 1: take few snapshots (3-5) in real-time of a target face 
or 
step 1 bis: use face imgs already taken from a target (label = true)

step 2: take a new snapshot (frame) in real-time as unlabelled img

step 4: compute bounding boxes using Viola-Jones algorithms to detect faces in every imgs

step 5: extract features for each img using PcaNet

step 6: compute one row of similarity matrix using L1 distance

step 7: solve the following optimization problem:

<p align="center">
  <img src="img/filters_l1.png" width="25%">
</p>

by producing the harmonic solution:

<p align="center">
  <img src="img/filters_l1.png" width="25%">
</p>

infer label and display on current frame.

## Results

Before going for real time inference we experimented on stored imgs.  

We used roughly 50 imgs 

<p align="center">
  <img src="img/filters_l1.png" width="25%">
</p>


## Limits

* Viola does not capture well changes of pose therefore to detect a face we need it to be mostly frontal

* we do not use a pre-trained segmentation tool to extract faces from background in bounding boxes.

