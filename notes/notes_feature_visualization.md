## Feature Visualization | [Paper](https://arxiv.org/abs/1706.07979) | [Notes](notes_feature_visualization.md) 
***
### Feature Visualization by Optimization
Feature vizualization by optimization works by starting with a random input and slowly tweaking the input until a certain structure of the DNN reaches its maximal possible state. Based on the objective at hand, we can optimize for a neuron, channel, layer, class logits or class probability. Optimizing pre-softmax logits tends to work better than the closs probabilities and leads to results of better visual quality.
***
### Diversity
Well-trained DNNs are often diverse in the sense that multiple relatively different instances of a certain class would elicit the same response. However, a single optimization run would end up with a single extremely positive input. In order to address the intra-class diversity, we can add an extra *diversity* term to our objective that pushes examples to differ from one another (e.g. penalize cosine similarity). However, we often end up with inputs which represent a strange mixture of different objects, due to the fact that neurons exist which are represent multiple concepts at the same time.
***
### Interaction between Neurons
Due to the problem defined in the previous section, a need arises to optimize for a combination of neurons, rather than a single one. In order to do so, we can define the activation space to be all possible combinations of neuron activations. A single activation would then be a basis vector in this space. A combination of neurons would just be a vector residing in the activation space. By using this as a framework, we can generate new images which represent a combination of neurons rather than a single one.
***
### The Enemy of Feature Visualization
If we want to visualize features by just optimizing an image to make neurons react without any constraints, we often end up with an noisy image full of nonsensical high-frequency patterns, which could be explained by the CONV & POOL nature of the majority of DNNs used in CV. In order to combat this problem, some form of **regularization** should always be provided. The forms of regularization can be categorized in three separate families. **Frequency penalization** is targeted at removing the high frequency noise from images. It could either penalize variance between neighbouring pixels or blur the image after each optimization step. **Transformation robustness** tries to find only this inputs which would elicit the same response even upon small variations in them. Here, a jitter, rotatation or scaling of the image before each optimization step could be used. Instead of applying heuristics we can also use **learned priors**, where we either learn a generator function such as GAN or VAE or learn a prior giving us access to the gradient of probability.
***
### Preconditioning and Parameterization
**Preconditioning** in optimization refers to reducing high frequencies in the gradient. It does not change the minimums, but rather guides the optimization process to certain minimums which might be more favourable. Using different reguliriztion metrics will thus lead to different results. The Linf metric increases the high frequencies, while the decorrelated space decreases them. Furthermore, the decorrelated space often leads to images of better quality and makes the optimization process faster.
***
### Q's
- Are the optimization with diversity examples a result of multiple runs of the same algorithm with a different starting point?
- In the third section, we introduce an activation space in order to provide a framework for combining multiple neuron activations. Consequently, the argumentation comes that the basis vectors (single neuron activations) be more interpretable than random directions (combination of neuron activations). If that is so, why do we need to consider the activation space in the first space?
- What is the decorrelated space refering to?