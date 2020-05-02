## [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
***
### Preliminaries
The understanding of a DNN can be either mechanical or functional. The mechanical (algorithmical) understanding refers to the inner workings of a DNN. The **functional understanding** concerns the characterization of the behavior of the DNN. A functional understanding could be done either through a **model analysis** or a **decision analysis**.\
The basic tools that are needed are interpretation and explanation. **Interpretation** is the mapping of an abstract concept (e.g. a vector space) in a human-understandable domain (e.g. an image/text/sound). **Explanation** refers to the collection of features of the interpretable domain that have contributed for a given input to produce a certain output (decision).
***
### Interpreting a DNN Model
DNNs make their decision based on *concepts* which they learn during the training phase. In the case of classification, DNNs have to learn a *prototype* representation for each of the predefined classes. In the inference phase, DNNs try to match the novel input to each of the available prototypes. The one which results in the best fit will also produce the highest score. Interpreting the DNN model then means that we need to find the corresponding protype for each of the defined class and map it in a human-understandable domain. This can be done through **Activation Maximization (AM)**. There are three types of such algorithms, which are **General AM**, **AM with an expert**, **AM in the code space (decoding function)**.\
The **AM algorithm** is a framework that finds an input that will produce the highest score for the concept (class) of interest. In it's general form, AM can be executed by maximizing the sum of the output from the DNN and the negative L2 regularization term of the input with gradient ascent. Due to the L2 regularization term generated images are generally gray with few edge and color paterns at specific locations.\
In the **AM with an expert** algorithm, the L2 regularization term is changed with a data density model *log p(x)*. Based on that, the generated prototypes will resemble more the actual training data and thus, carry more interpretation power in the human-understandable domain. A possible choice for the data model is a Gaussian or convolutional RBM (Restricted Boltzmann Machine). An overfitted *expert* might lead to the fact that the generated input represents more the training data distribution as a whole, rather than the actual learned concept *Wc*. An underfitted *expert* will lead to a natural looking but unlikely generated input which also won't be a truly representative of *Wc*.\
Due to the difficulty of training a good *expert*, another possibility arises in the face of **AM in the Code Space**. Here a second generative model is trained which tries to recreate the input based on a given output from the original DNN. The model is trained by maximizing the *log p(Wc|g(z))* which represents the classification of the output from the generative model by the original DNN.\
However, in practice both *p(Wc|x)* and *p(x)* might be multimodal leading to the fact that no single prototype exists which truly represents the learned concept *Wc*. Based on that, it makes more sense to ask the question *What are the features of the input that make it a representative of the concept Wc?*, rather than trying to find a single prototype which explains the learned concept.\
***
### **Explaining DNN Decisions**
In order to find the features which drive the decions of the DNN, the following methods can be applied.\
**Sensitivity analysis** tries to answer the question *What makes x more/less Wc?* Here the relevance scores *Ri* are equal to the gradient squared.The most relevant *neurons* are thus the ones for which the output is most sensitive.\
**Simple Taylor Decomposition** tries to explain the decision of the model by decomposing the output value as a sum of relevance scores *Ri*. The relevance scores are computed by multiplying the *sensitivity* (gradient) with the *saliency* (input) of the neuron.\
**Relevance propagation** decomposes a DNN by starting at the output and redistributing the prediction score going backwards. The redistribution of the score should be **conserved**. \
Further techniques are *guided backprop* & *deconvolution*. Out of all discussed techniques, only LRP supports both pooling and filtering
***
### LRP Explanation Framework
The reference scores are calculated by the formula below. The weights and activations are divided in two terms based on the sign of the weights. The sum of positive weights is multiplied by the factor alpha and the negative weights are multiplied by the factor beta. The used LRP is defined by the two constants. Popular choices are LRP-a1b0 and LRP-a2b1.
![LRP Formula](./assets/understanding_form_1.png)
The LRP algorithm can further be interpreted as a Taylor decomposition. Due to the iterative nature of the mechanism, the name *Deep Taylor Decomposition* has been adopted. The *Deep Taylor LRP* rules for more traditional layer types can be found in the paper, while the handling of more special layers is not agreed upon in the literature.
***
### Recommendations and Tricks for LRP
- LRP works best for CONV ReLU netwworks which don't have excessive ammount of DENSE layers.
- After every DENSE layers the usage of *dropout* is recommended.
- Sum-pooling leads to better results than max-pooling.
- Biases should be zero or negative at training time.
- LRP-a1b0 should be used as a standard. If negative relevance is needed, LRP-a2b1 can be used in the hidden layers.
- In case of unsatisfactory heatmaps, a larger set of propagation rules is to be defined.
- In case of noisy first-layer filters, large stride in the first CONV layer or not optimally trained classifiers, a translation trick should be applied in order to improve the generated heatmaps.
- The relevant patern can be visualized by multiplying the input image with the generated *relevance mask*, the values of which should be normalized between 0 and 1.
***
### Quantifying Explanation Quality
**Explanation Continuity** means that similar input data points should have similar explanations. Both sensitivity analysis and the simple Taylor decomposition are discontinuous on the x1 = x2 line, while the deep Taylor LRP is not. \
**Explanation Selectivity** means that we iteratively remove the features of the input with the highest relevance score. The *certainty* of the classification should then, as a result of that, drop sharply. Deep Taylor LRP performs the best out of the three methods.
***
### Q's:
- Is the *expert* in the AM algorithm pretrained?
- Can we not use a GAN-like approach in the case of AM in Code Space? If yes, does it not make sense to train the original model in such a fashion so that we have interpretation embedded in the original training process?
- Should we dive deeper into the Taylor series explanation of the LRP?
- Do filtering and pooling in Table 1 mean the techniques for CNNs or something else?
- What is the function f in the sliding window trick?