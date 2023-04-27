# BoundaryDiffusion

This is the official Pytorch implementation of the paper **[Boundary Guided Mixing Trajectory for Semantic Control with Diffusion Models](https://arxiv.org/abs/2302.08357)**.


## 1. Project Overview
In this work, we present a **learning-free** method for applying pre-trained Denoising Diffusion Models (DDMs) on semantic control and image manipulation in **one single-step operation**. Speficically, we propose to guide the denoising trajectory to across the target semantic boundary to achieve the image editing purpose. It is worth noting that the semantic boundaries are formed during the training process of diffusion models, and our method **do not need to fine-tune or learning any extra editing neural network modules**, allowing for very efficient and light-weighted downstream applications.

<p align="center">
    <img src="assets/trajectory.png" width="500">


## 2. Theoretical Analysis of Diffusion Models

Our methodology design in this work is based on the analysis of high-dimensional latent spaces of the pre-trained denoising diffusion models. Specifically, we propose to study the probablistic and geometric properties of latent spaces given differnt sources of latent encodings (i.e., sampling vs. inversion), and observe that the inverted latent encodings do not follow the standard Gaussian distribution in the departure latent space as for the directly sampled ones.


 <p align="center">
    <img src="assets/geometry.png" width="500">


Our theoretical analysis also introduces the concept of **Mixing Step** to characterize the convergence of pre-trained diffusion models, inspired by the **Markov mixing time** study.
Please refer to our paper for more details.
