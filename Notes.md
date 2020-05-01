# [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
***
## Preliminaries
The understanding of a DNN can be either mechanical or functional. The mechanical (algorithmical) understanding refers to the inner workings of a DNN. The **functional understanding** concerns the characterization of the behavior of the DNN. A functional understanding could be done either through a **model analysis** or a **decision analysis**.
The basic tools that are needed are interpretation and explanation. **Interpretation** is the mapping of an abstract concept (e.g. a vector space) in a human-understandable domain (e.g. an image/text/sound). **Explanation** refers to the collection of features of the interpretable domain that have contributed for a given input to produce a certain output (decision).
***
## Interpreting a DNN Model
DNNs make their decision based on *concepts* which they learn during the training phase. In the case of classification, DNNs have to learn a *prototype* representation for each of the predefined classes. In the inference phase, DNNs try to match the novel input to each of the available prototypes. The one which results in the best fit will also produce the highest score. Interpreting the DNN model then means that we need to find the corresponding protype for each of the defined class and map it in a human-understandable domain. This can be done through **Activation Maximization (AM)**. There are three types of such algorithms, which are **General AM**, **AM with an expert**, **AM in the code space (decoding function)**.
The **AM algorithm** is a framework that finds an input that will produce the highest score for the concept (class) of interest. In it's general form, AM can be executed by maximizing the sum of the output from the DNN and the negative L2 regularization term of the input with gradient ascent. Due to the L2 regularization term generated images are generally gray with few edge and color paterns at specific locations.
In the **AM with an expert** algorithm, the L2 regularization term is changed with a data density model *log p(x)*. Based on that, the generated prototypes will resemble more the actual training data and thus, carry more interpretation power in the human-understandable domain. A possible choice for the data model is a Gaussian or convolutional RBM (Restricted Boltzmann Machine). An overfitted *expert* might lead to the fact that the generated input represents more the training data distribution as a whole, rather than the actual learned concept *Wc*. An underfitted *expert* will lead to a natural looking but unlikely generated input which also won't be a truly representative of *Wc*.
Due to the difficulty of training a good *expert*, another possibility arises in the face of *AM in the Code Space*. Here a second generative model is trained which tries to recreate the input based on a given output from the original DNN. The model is trained by maximizing the *log p(Wc|g(z))* which represents the classification of the output from the generative model by the original DNN.
However, in practice both *p(Wc|x)* and *p(x)* might be multimodal leading to the fact that no single prototype exists which trully represents the learned concept *Wc*. Based on that, it makes more sense to ask the question *What are the features of the input that make it a representative of the concept Wc?*, rather than trying to find a single prototype which explains the learned concept.
***
## **Explaining DNN Decisions**
In order to find the features which drive the decions of the DNN, the following methods can be applied.
**Sensitivity analysis** tries to answer the question *What makes x more/less Wc?* Here the relevance scores *Ri* are equal to the gradient squared
**Simple Taylor Decomposition** tries to explain the decision of the model by decomposing the output value as a sum of relevance scores *Ri*. The relevance scores are computed by multiplying the *sensitivity* (gradient) with the *saliency* (input) of the neuron.
***
## LRP Explanation Framework


***
# CVPR18 Interpreting and Explaining Deep Models in Computer Vision

- Types of understanding of NN:
    - Mechanic understanding
    - Functional understanding
        - Model analysis
        - Decision analysis

- Approaches
    - Class Prototypes
        - How does a class look to a NN?
        - argmax f(x) + regularization
    - Individual explanations:
        - Why is a given image classified as a sheep?
    - Sensitivity analysis
        - The relevance of input features given by the squared partial derivative
        - Sensitivity analysis explains a variation of the function, but not the function itself
        - If you were to change a pixel, would it be more like a car or not. Which pixels should I change to make it look like a car?
        - Problem: Shattered gradients
    - Layerwise Relevant propagation
        - not based on Gradient X Input!
        - Which pixels contribute how much to the classification?
        - 