# Honours

This branch contains most of the code used for the first phase of the Project. 
The primary goal was to verify the capability of GNNs to learn and predict the dynamics of spring mass systems.
I primarily focused on spring-mass chains.

There are two main types of chains that we learn to simulate:
 - Two-Fixed end chains: Where the first and last masses of the chains are immovably fixed to the background.
 - One-Fixed end chains: Where only one of the masses are fixed.
 
 In addition, we also experimented with damping ratios, different spring constants, and architectural changes to the model we use.
 
To use this, you need Tensorflow 1.x with CUDA setup for GPU accelerated training.
