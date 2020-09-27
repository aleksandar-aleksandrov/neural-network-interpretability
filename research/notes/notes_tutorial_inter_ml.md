## Tutorial on Interpretable Machine Learning | [Paper](http://heatmapping.org/slides/2018_MICCAI.pdf) | [Video](https://www.youtube.com/watch?v=LtbM2phNI7I&t=2078s) | [Notes](notes_tutorial_inter_ml.md) 
***
### Overview
Different techniques: **Sensitivity**, **Deconvolution**, **LRP**, and **Friends**  
Perspectives to understand Deep Nets:  
  * Mechanistic Understanding: Understand how each neuron works 
  * Functional Understanding: Understand how the network maps the input to output
     * Model Analysis: In a view of whole ensemble of the data (average)
     * Decision Analysis: In a single decision view (individual)
***
### Model Analysis
1. Class Prototypes  
  How does the appropriate class prototype look like for the whole ensemble. *argmaxf(x)* to find the typical look according to the neural network.

***
### Decision Analysis
1. Sensitivity Analysis  
  Analyze neural network by computing the partial derivative of the  overall function w.r.t xi (gradient of every **pixel**).  
  Q: Which pixels lead to increase/decrease of predicition score when **changed**?  
  Problems: 
   * Sensitivity analysis doesn't highlight pixels corresponding to cars -> Explain the variation of the function not the function itself (e.g. Highlight pixels make the image more or **less** a car)  
   * Shattered Gradients: Input gradient becomes increasingly highly varying and unreliable with neural network depth
2. Individual Explanations (LRP)  
  Why is a **given image** classified as a car? *heatmap = LRP(x,f)*  
  Q: Which pixels **contribute how much** to the classification?  
  LRP resolve the shattered gradients issue from not computing **gradient x inputs**
***
### LRP
Steps  
  1. Initialize relevance of last layers *rj* to results of the classification *rj = f(x)*
  2. Compute the relevance from uppper layers using simple LRP rule
  3. Relevance Conservation Property: Sum of relevance in every layers is constant (*f(x)*). 
  
Axiomatic Approach to interpretablity: Ground truth (Highlighting pixels that deep neural network comes up for car) is hard to obtain for multiple classes -> Evaluate the explanation technique **axiomatically** (e.g. It must pass a number of predefined **unit tests**)  
Properties of Heatmap  
  1. Conservation: Total attirbution on the input features should be proportional to the amount of (explainable) evidence at the output. Use LRP-a1b0 propagation rule to prove.
  2. Positivity: If the neural network is certain about its prediction, input features are either relevant (positive) or irrelevant (zero)
  3. Continuity: If two inputs are almost the same(a image of car and a image of car with slight perturbation), and the prediction is almost the same, then the explanation should also be almost the same. On the other hand, **Gradient based methods** (e.g. Sensitivity Analysis) produce discontinuous explanation, because their heatmaps tend to flicker at much higher frequency than images are actually changing.
  4. Selectivity: If input features are attributed relevance, removing them should reduce evidence at the output (Test selectivity with Pixel-Flipping)  
  
LRP-a1b0 has all these four properties
***
### From LRP to Deep Taylor Decomposition
The LRP-a1b0 can be seen as a deep Taylor decomposition (DTD) which yields domain- and layer- specific propagation rules.  
Proposition: Relevance at **each layer** which is propogated from the top layer to the input layer is a product of the activation and an approximately constant term.  
1. Build the Relevance Neuron: Relevance *Rj* can be seen as a function of the previous layer. *cj* modulation term which is constant and positive.   
2. Expand the Relevance Neuron: Analyze the previous max function (relu) using Taylor decomposition (near zero of the relu function), which will give us what is the proper weight to redistribute from one neuron to the neuron of the previous layer. Relevance as a fucntion of the  activation in the previous layer = relevance evaluated in the root point + a sum of the first oder Taylor expansion terms + epsilon.
3. Decompose Relevance: *Rj(root point)=0* because the root point is near zero. The middle term is the **sum of all input features** -> Identify what are the importance of each input neurons for producting relevance in the higher layer  
How to choose the root point  
   1. nearest root: Descend along the direction of gradient of the relevant neuron until hit the root point
   2. rescaled activation: Decend along the direction of activation which is rescaled towards the origin
   3. **rescaled excitations**: Following ii. + Descend in the input domain only along directions that have positive weights -> Gives the generic LRP-a1b0 rule. Root point belongs to the input domain of activation function + root point is close to the actual activation pattern.
4. Application to input layers: 
   1. Choose a root point that is nearby and satisfies domain constraints
   2. Inject it in the generic DTD rule to get the specific rule
5. Pooling relevance over all outgoing neurons: Treat pooling layers as relu detection layers

General pipeline for CNN: **Add relu at the last layer** + **use LRP-a1b0 whether its pulling or convolution** + **DTD for pixels at the first layer**
***
### QA
* I suggest we have a discussion about the Deep Taylor Decomposition as this is the most abstract concept among all the papers.
