# Why: Causal Inference Library

we present ”Why”, a causal inference library for structural causal modeling and identification. ”Why” implements a variety of algorithms, including the pc algorithm for causal discovery and multiple independence tests such as pearsons correlation, partial correlation, and chi squared. We also introduce the use of Generative Neural Networks (GNNs) and Causal Generative Neural Networks (CGNNs) for orienting the edges of structural causal models. Additionally, ”Why” offers bound estimation for the average treatment effect (ATE) under various assumptions, as well as sensitivity analysis for the presence of unobserved linear confounders. Our library also includes implementations of backdoor adjustment and ATE estimation using the COM, GCOM, and TARNet methods. Overall, ”Why” provides a comprehensive toolkit for causal inference in a range of settings.

# Documentation
An explantation for the methods implemented can be found in the pdf [Documentation, the Why library](https://github.com/MRSAIL-Mini-Robotics-Software-AI-Lab/Why/blob/main/Documentation%2C%20the%20Why%20library.pdf)
## Example Usage
An example on how to use this library can be found in the following notebooks:
* lucas.ipynb
* ihdp.ipynb
* pubg.ipynb