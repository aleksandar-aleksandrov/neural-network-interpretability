# Interpretability of Neural Networks

**This project has been done in collaboration with [@hans66hsu](https://github.com/hans66hsu) as part of the Master Practical Course "Beyond Deep Learning: Uncertainty-Aware Models" at the Technical University of Munich.**


## I. Project work overview

Our complete work can be found in this repository. Our weekly reports which track our step-by-step progress and our final report can be found in the `report` folder. The two interim and the final presentations can be found in the `presentation` folder. Our implementations are residing in the `implementation` folder. We have a jupyter notebook for every interpretation method that we have implemented. Some of the methods are showcased together in the same jupyter notebook to make their comparison easier. The full implementation can be found in the `nn_interpretability` package which resides in the folder with the same name. The models that we have trained and that we are using for experimentation purposes can be found in the `models` subfolder of the `implementation` folder. Our literature research is compiled in the `research` folder. We have written detailed notes of every paper we have reviewed together with our results for the corresponding technique. An exhaustive list of the papers we have reviewed and implemented can be found in the next section of this README.

## II. Literature we have reviewed
All research resources that we have found can be found [here](./research/README.md). In the following list, a comprehensive compilation of all the reviewed papers together with our notes and implementations are to be found.

- [X] Methods for Interpreting and Understanding Deep Neural Networks | [Paper](https://arxiv.org/abs/1706.07979) | [Notes](./research/notes/notes_methods_from_interpreting_dnn.md) | [Implementation I](./implementation/1.Activation_Maximization.ipynb) |  [Implementation II](./implementation/8.1.LRP.ipynb) | [Implementation III](./implementation/8.2.LRP_Transpose.ipynb)
- [X] Feature Visualization | [Paper](https://distill.pub/2017/feature-visualization/) | [Notes](./research/notes/notes_feature_visualization.md) 
- [X] The Building Blocks of Interpretability | [Paper](https://distill.pub/2018/building-blocks/) | [Notes](./research/notes/notes_building_blocks.md) 
- [X] Tutorial on Interpretable Machine Learning | [Paper](http://heatmapping.org/slides/2018_MICCAI.pdf) | [Notes](./research/notes/notes_tutorial_inter_ml.md)
- [X] Visualising image classification models and saliency maps | [Paper](https://arxiv.org/pdf/1312.6034.pdf) | [Notes](./research/notes/notes_visualizing_models.md) | [Implementation](./implementation/3.Saliency_Maps.ipynb)
- [X] Visualizing and understanding convolutional networks | [Paper](https://arxiv.org/pdf/1311.2901.pdf) | [Notes](./research/notes/notes_visualize_&_understand.md) | [Implementation](./implementation/4.Deconvolution.ipynb) | [Implementation III](./implementation/5.Occlusion_Sensitivity.ipynb)
- [X] Striving for Simplicity: The All Convolutional Net | [Paper](https://arxiv.org/pdf/1412.6806.pdf) | [Notes](./research/notes/notes_striving_for_simplicity.md) | [Implementation](./implementation/6.Backpropagation.ipynb)
- [X] Axiomatic Attribution for Deep Networks | [Paper](https://arxiv.org/pdf/1703.01365.pdf) | [Notes](./research/notes/notes_axiomatic_for_dnn.md) | [Implementation](./implementation/6.Backpropagation.ipynb)
- [X] Layer-Wise Relevance Propagation: An Overview | [Paper](http://iphome.hhi.de/samek/pdf/MonXAI19.pdf) | [Notes](./research/notes/notes_lrp_overview.md) | [Implementation](./implementation/8.1.LRP.ipynb)
- [X] SmoothGrad: removing noise by adding noise | [Paper](https://arxiv.org/pdf/1706.03825.pdf) | [Notes](./research/notes/notes_smoothgrad.md) | [Implementation](./implementation/6.Backpropagation.ipynb)
- [X] Learning Deep Features for Discriminative Localization | [Paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) | [Notes](./research/notes/notes_cam.md) | [Implementation](./implementation/10.1.Class_Activation_Map.ipynb)
- [X] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | [Paper](https://arxiv.org/pdf/1610.02391.pdf) | [Notes](./research/notes/notes_grad_cam.md) | [Implementation](./implementation/10.2.Grad_Class_Activation_Map.ipynb)
- [X] Learning Important Features Through Propagating Activation Differences
 | [Paper](https://arxiv.org/pdf/1704.02685.pdf) | [Videos](https://www.youtube.com/watch?v=v8cxYjNZAXc&list=PLJLjQOkqSRTP3cLB2cOOi_bQFw6KPGKML) | [Notes](./research/notes/notes_deeplift.md) | [Implementation](./implementation/9.DeepLIFT.ipynb)
- [X] Inceptionism: Going Deeper into Neural Networks | [Paper](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) | [Notes](./research/notes/notes_inceptionism.md) | [Implementation](./implementation/2.Deep_Dream.ipynb)
- [X] On Calibration of Modern Neural Networks
 | [Paper](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf) | [Notes](./research/notes/notes_temperature_scaling.md) | [Implementation](./implementation/13.Uncertainty_Aware_DeepLIFT.ipynb)

## III. Package overview
Our main deliverable for this project is the package `nn_interpretability`, which entails every implementation of a NN interpretability method that we have done as part of the course. It can be installed and used as a library in any project. In order to install it one should navigate to the `implementation` folder and execute the following command:
```
pip install -e .
```
After that, the package can be used anywhere by importing it:
```
import nn_interpretability as nni
```
Next to the interpretability functionality, we have defined a repository of used models and additional functionality for loading and visualizing data and the results from the interpretability methods. Furthermore, we have implemented uncertainty techniques to aid the behavior of interpretability methods under stochastical settings.
Every implemented interpretability method or technique have an accompanying Jupyter Notebook as outlined in the next section of this README. We have, also, prepared a general demonstration of the developed package in this [Jupyter Notebook.](./implementation/14.Demo.ipynb)

## IV. What have we implemented?
### 1. Model-based approaches
 - Activation Maximization
   - [X] General Activation Maximization | [Jupyter Notebook](./implementation/1.Activation_Maximization.ipynb)
   - [X] Activation Maximization in Codespace (GAN) | [Jupyter Notebook](./implementation/1.Activation_Maximization.ipynb)
   - [X] Activation Maximization in Codespace (DCGAN) | [Jupyter Notebook](./implementation/1.Activation_Maximization.ipynb)
 - DeepDream | [Jupyter Notebook](./implementation/2.Deep_Dream.ipynb)

### 2. Decision-based approaches
 - Saliency Map | [Jupyter Notebook](./implementation/3.Saliency_Maps.ipynb)
 - DeConvNet
   - [X] Full Input Reconstruction | [Jupyter Notebook](./implementation/4.Deconvolution.ipynb)
   - [X] Partial Input Reconstruction | [Jupyter Notebook](./implementation/4.Deconvolution.ipynb)
 - Occlusion Sensitivity | [Jupyter Notebook](./implementation/5.Occlusion_Sensitivity.ipynb)
 - Backpropagation
   - [X] Vallina Backpropagation | [Jupyter Notebook](./implementation/6.Backpropagation.ipynb)
   - [X] Guided Backpropagation | [Jupyter Notebook](./implementation/6.Backpropagation.ipynb)
   - [X] Integrated Gradients | [Jupyter Notebook](./implementation/6.Backpropagation.ipynb)
   - [X] SmoothGrad | [Jupyter Notebook](./implementation/6.Backpropagation.ipynb)
 - Taylor Decomposition
   - [X] Simple Taylor Decomposition | [Jupyter Notebook](./implementation/7.Taylor_Decomposition.ipynb)
   - [X] Deep Taylor Decomposition | [Jupyter Notebook](./implementation/7.Taylor_Decomposition.ipynb)
 - LRP
   - [X] LRP-0 | [Jupyter Notebook](./implementation/8.1.LRP.ipynb)
   - [X] LRP-epsilon | [Jupyter Notebook](./implementation/8.1.LRP.ipynb)
   - [X] LRP-gamma | [Jupyter Notebook](./implementation/8.1.LRP.ipynb) 
   - [X] LRP-ab | [Jupyter Notebook](./implementation/8.1.LRP.ipynb)
   - [X] LRP Transpose | [Jupyter Notebook](./implementation/8.2.LRP_Transpose.ipynb)
 - DeepLIFT
   - [X] DeepLIFT Rescale | [Jupyter Notebook](./implementation/9.DeepLIFT.ipynb)
   - [X] DeepLIFT Linear | [Jupyter Notebook](./implementation/9.DeepLIFT.ipynb)
   - [X] DeepLIFT RevealCancel | [Jupyter Notebook](./implementation/9.DeepLIFT.ipynb)
 - CAM
   - [X] Class Activation Map (CAM) | [Jupyter Notebook](./implementation/10.1.Class_Activation_Map.ipynb)
   - [X] Gradient-Weighted Class Activation Map (Grad-CAM) | [Jupyter Notebook](./implementation/10.2.Grad_Class_Activation_Map.ipynb)

### 3. Uncertainty
 - Monte Carlo Dropout 
   - [X] Monte Carlo Dropout Analysis | [Jupyter Notebook](./implementation/11.MC_Dropout_Interpretability.ipynb)
   - [X] Uncertainty interpretability with LRP | [Jupyter Notebook](./implementation/11.MC_Dropout_Interpretability.ipynb)
 - Evidential Deep Learning
   - [X] Evidential Deep Learning Anaylsis | [Jupyter Notebook](./implementation/12.Evidential_Interpretability.ipynb)
   - [X] Base Model vs. Evidential Deep Learning Model with LRP | [Jupyter Notebook](./implementation/12.Evidential_Interpretability.ipynb)
 - Uncertain DeepLIFT
   - [X] DeepLIFT Deterministic vs. Stochastic Model | [Jupyter Notebook](./implementation/13.Uncertainty_Aware_DeepLIFT.ipynb)
   - [X] DeepLIFT Random Noise | [Jupyter Notebook](./implementation/13.Uncertainty_Aware_DeepLIFT.ipynb)
   - [X] Temperature scaling | [Jupyter Notebook](./implementation/13.Uncertainty_Aware_DeepLIFT.ipynb)
