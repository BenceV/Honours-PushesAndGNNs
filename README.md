# Honours: Phase 3

The final part of the project with a substantial amount of rework done to the code base. 

In this phase, we moved away from the spring-mass simulation based training to an alternative setup where the model is trained on real world examples extracted from a publicly available dataset. This dataset, curtesy of MIT - Cubelab, is called The Push dataset, and it contains a massive set of object end-effector interactions.

These trajectories, are not by default applicable for the GNN, therefore a lot of preprocessing was required. Much of this preprocessing is done offline. 
The code can be found on https://github.com/BenceV/MITPushProcessor for this processing step. 

In addition to the new dataset, there was also a lot of changes in the way we calculate errors, losses, train the model, evaluate performance.
We moved to use Tensorflow 2, which required a lot of changes as our model and pipeline was completely custom.
