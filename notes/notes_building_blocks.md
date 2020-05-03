## The Building Blocks of Interpretability | [Paper](https://distill.pub/2018/building-blocks/) | [Notes](./notes/notes_building_blocks.md) 
***
### Making Sense of Hidden Layers
While the input and output layers are often easy to understand and explain, the hidden layers encapsulate intermediate abstract concepts presented in vector spaces based on which the DNN makes decisions. Every such layer can be seen as a 3D cube the cells of which represent activations. The x and y coordinates correspond to positions in the input, while the z coordinate refers to a channel/detector being run on the input.\
In order to make sense of the abstract mathematical space, we build **semantic dictionaries**. To do so, we map neuron activations to a visualization of the neuron and sort them by magnitude. Such dictionaries build the base for the interpretability of DNN.
***
### What Does the Network see?
Building up on the semantic dictionaries, we can combine multiple neurons and their visualizations to reason about the interpretation of the input from the DNN in a region of the input. By doing this iteratively, we deduce that deeper layers can find more sophisticated structures in the input.
***
### How Are Concepts Assembled?
While the visualization shows what a DNN can detect, they still do not provide any information about the way DNNs take decisions. The most common interface for attribution is called a **saliency map**, which highlights the regions of the input which contribute the most to the final decision of the DNN. This method exhibits two problems. First, pixels are often entangled with other pixels and provide little connection to the high-level concepts. Secondly, they don't allow for probing into the hidden layers. In order to improve on that, we can apply the method not only to the input, but also to the hidden layers to understand the gradual conceptualizing of classes that the DNN undertakes.
\
An alternate method is the **channel attribution**, by which we slice the cube by channels rather than spatial positions, so that we can find out the role of each filter/channel/detector in the final decision.
***
### Making Things Human-Scale
One of the main issues that arises by trying to provide a human-understandable interpretation of DNNs is the sheer scale of modern DNNs, which normally consist of millions of neurons spread across tens or even hundreds of layers, each of which consisting of hundreds of channels. Trying to implement previously discussed techniques leads to enormous amount of information that an user should be able to follow in order to understand a single decision of a DNN. In order to prevent that, we can group neurons together by using matrix factorization in order to make the understanding of the higher-level concepts that the DNN has learned easier for a human. These groups can spawn in both spatial, but also channel direction. These groups will then be the atomar structures on the base of which interfaces for the interpretation of DNNs could be constructed.
***
### The space of interpretability Interfaces
An interface is an union of elements, which display a specific type of content (e.g. activations or attribution) using a corresponding style of presentation (e.g. feature visualization). The content of elements lives in a substrates defined by the decomposition of layers into atoms (e.g. neurons, channels, groups, etc). Each interface should be constructed in such a way that it answers what the DNN recognizes, how it develops understanding or focus on making the things human-scale. \
Furthermore, the provided set of building blocks is not exhaustive and can be further extended by other dimensions such as the role of the dataset or the model parameters. Such interfaces could also be used for comparing models or seeing the evolution of the decision-making process during the learning phase of the DNN.
***
### How Trustworthy Are These Interfaces?
Based on earlier research, it can be concluded that directions in DNN are semantically meaningful. Furthermore, neurons have been found to respond to a mixture of concepts, rather than a single one. A concern in the case of attribution is the **"path-dependance"**, which, however, turns out to not have a significant influence on the output. In general, it is noteworthy to say, that further investigation in the applicability of methods in the domains of safety and fairness needs to be done.
***
### Q's
- How are spatial and channel attribution different? Does the spatial one only concerns itself with a single channel or positions across all channels?
