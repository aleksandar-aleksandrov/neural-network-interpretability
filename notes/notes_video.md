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