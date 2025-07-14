## Neural Spline Flow for 1 dimension probability density estimation


The easiest way to get started with this code is by checking out the notebook [`tutorial.ipynb`](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/tutorial.ipynb). This tutorial outlines the process of estimating the distribution of potential outcomes for a specific treatment option by first clustering the potential outcome dataset based on each patient's tumor volume at previous time points. Patients are grouped according to the severity of their condition, and a specific patient subgroup is then selected for detailed analysis.  

This is a PyTorch implementation of Neural Spline Flow for 1D distributions, extended from the original paper "Neural Spline Flows" by Conor Durkan et al.  

The example dataset used in this repository is data/processed_potential_outcomes.csv, which contains the potential outcomes of cancer volume under different treatment options:  

1: Not treated  

2: Chemotherapy only  

3: Radiotherapy only  

4: Chemotherapy + Radiotherapy  

In the example below, I implemented a toy distribution to verify that 1D Neural Spline Flow can still capture complex distributional shapes. The number of bins is set to 10 for all experiments. The learning rate, number of iterations, and other hyperparameters are defined in the script file `nf_spline.sh`.
To reproduce the results, simply set the hyperparameters in `nf_spline.sh` and run the script.  

These toy experiments demonstrate the flexibility of normalizing flows by showing how they can transform a standard Gaussian into various complex target distributions, including skewed distributions, mixtures, and sinusoidal patterns with noise.

Animations of the learned potential outcome distributions are included to visualize how the normalizing flow gradually expands and contracts the input space to match the desired target density. By feeding samples from a standard Gaussian into the trained flow network, we can generate new samples from the target distribution.


| Skewed | Mixture | Sin+Noise |
|--------------|-------------------|-----------|
|     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/skewed_distribution.jpg "Density $skewed$") |![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/Mixture_gaussian.jpg "Density $mixture$") | ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/img/sinnoise.jpg "Density $sinnoise$")      |
  

| Control (Not treated) | Treatment (Chemo+Radio) | Result |
|--------------|-------------------|-----------|
|     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/gifs/Cancer_po_treatment_1_cluster_2.gif "Density $skewed$") |![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/gifs/Cancer_po_treatment_4_cluster_2.gif "Density $mixture$")                |     ![alt text](https://github.com/Jeong-Eul/NeuralSplineFlow-1D/blob/main/code/Cancer_PO_distribution/po_treatment_4_cluster_2_marginal_distribution/marginal_distribution.png "Density $sinnoise$")      |
  
