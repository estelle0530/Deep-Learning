# GRU-based multi head model to predict microbe-host and microbe-microbe interactions

## Background
The human microbiome is a dynamic system of trillions of organisms interacting with each other and their human host. These microbe-microbe and host-microbe interactions evolve in response to perturbations like changes in diet or treatment by drugs and antibiotics. Understanding these changes can provide insights into how to shape the microbiome to benefit the host. In light of the nonlinear relationship among microbes, recent research has increasingly utilized artificial neural networks (ANN) to model microbial communities. An advantage to using an ANN to model microbial dynamical change is theability to conduct predictive in silico experiments by artificially perturbing input data to predict the perturbation effect on the entire microbial composition. 

## Method
We proposed a GRU-based neural network to model the microbial composition and absolute abundance at each time point. We have two model types: one to model the microbial composition at each time point within a subset of classes and a second model to predict absolute abundance at each time point. Both models also classify whether the sample predicted is from a healthy or disease donor. Once our multi-head and multi-directional models are well-trained, determined by a sufficiently small MSE, we will conduct in-silico experiments by perturbing inputs for prediction.

![alt text](https://github.com/estelleyao0530/Deep-Learning/blob/main/Figure/gru_schematic.png)
