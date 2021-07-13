# Honours: Phase 2 

This branch contains the code base used for the second phase of the project. 
The primary goal of this phase was to adapt the architecture used in the first phase to allow the approximation and simulation of rigidbodies. 
Using this new system we were able to train our model to be able to estimate the future state of these simulated spring-mass rigidbodies. 

Inside the main folder for this branch there are 4 folders:
 - Moving_Rigidbody: Here we have all the code needed for experiments where the simulated rigid-body (Very rough approximation), is moving as it is being pushed by an end-effector.
 - Moving_Rigidbody_Network_sizes: In this folder, is the code needed for experimenting with different update function sizes (Each a small neural network), with a moving rigidbody setup.
 - Not_Moving_RigidBody: In this setup, the rigidbody is not being pushed by the end-effector, instead the end-effector acts as an anchor and fixes the rigidbody on one point to the background.
 - Omnipush: This folder contains some preliminary exploration done into the possible usage of the Omnipush Dataset, MIT. 


To use this, you need Tensorflow 1.x with CUDA setup for GPU accelerated training.
