## Neural Spline Flow for 1 dimension probability density estimation

The easiest way to get started using this code is probably by checking out the notebook [`tutorial.ipynb`](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/tutorial.ipynb).

Pytorch implementation of Neural Spline Flow 1D which is expanded the implementation the paper "Neural Spline Flows" by Conor Durkan, et al. 
The example dataset used in this protocol is predicted cancer potential outcome (processed_potential_outcomes.csv) which is potential outcome of cancer volume for assigned treatment option (1: not treated, 2: Chemo only, 3: Radio only, 4: Chemo + Radio)  

Below example, I have implemented some toy example distribution to identify 1D NeuralSplineFlow still estimate a complicate distribution shape. The number of bins are set to *10* for all experiments. The learning rate, number of iterations and other hyperparameter used are given by the experiment script file (nf_spline.sh). Set the hyperparameters of the normalising flow in `nf_spline.sh` and run to reproduce these results.  

The yoy experiments highlight the flexibility of normalising flows by showing that they can transform a standard Gaussian into skewed, mixture distribution and sin+noise.   

The animations of potential outcome distribution are added to show how the normalising flow gradually expands and contracts the input space into the desired target density. By feeding samples from a standard bivariate Gaussian into the trained flow network, we can draw new samples from the target density.  


| Skewed | Mixture | Sin+Noise |
|--------------|-------------------|-----------|
|     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/skewed_distribution.jpg "Density $skewed$") |![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/Mixture_gaussian.jpg "Density $mixture$")                |     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/sinnoise.jpg")      |
  

| Control (Not treated) | Treatment (Chemo+Radio) | Result |
|--------------|-------------------|-----------|
|     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/gifs/Cancer_po_treatment_1_cluster_2.gif "Density $skewed$") |![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/gifs/Cancer_po_treatment_4_cluster_2.gif "Density $mixture$")                |     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/Cancer_PO_distribution/po_treatment_4_cluster_2_marginal_distribution/marginal_distribution.png "Density $sin+noise$")      |
  
