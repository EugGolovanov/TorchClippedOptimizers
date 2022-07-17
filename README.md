# TorchClippedOptimizers

`torch-clip` a library to improve optimization methods by clipping off heavy-tailed gradient. This makes it possible to increase the accuracy and speed of convergence during the training of neural networks on a specific number of tasks.

#### `torch-clip` a library to improve optimization methods by clipping off heavy-tailed gradient. This makes it possible to increase the accuracy and speed of convergence during the training of neural networks on a specific number of tasks.
<br>*Example of the distribution of gradient lengths with heavy tails:*
![This is an image](readme_images/heavy_tail.jpg)
------------
### our clipping methods

+ [Linear Stoch Norm Clipping](#LinearStochNormClip);  
+ [Quadratic Stoch Norm Clipping](#QuadraticStochNormClip);  
+ [Layer Wise Clipping](#LayerWiseClip);  
+ [Coordinate Wise Clipping](#CoordWiseClip);  
+ [Auto Clipping](#AutoClip);  
+ [Linear Stoch Auto Clipping](#LinearStochAutoClip);  
+ [Quadratic Stoch Auto Clipping](#QuadraticStochAutoClip).



### Installation
you can install our library using pip:  
`pip install torch-clip`  


### What do you need us for?
In the last few years, for various neural network training models (for example, BERT + CoLA), it has been found that in the case of "large stochastic gradients", it is advantageous to use special clipping (clipping/normalization) of the patched gradient. Since all modern machine learning, one way or another, ultimately boils down to stochastic optimization problems, the question of exactly how to "clip" large values of patched gradients plays a key role in the development of effective numerical training methods for a large class of models. This repository implements optimizers for the pytorch library with different clipping methods.  



##### <a name="LinearStochNormClip"></a>	Linear Stoch Norm Clipping
about this clipping methods  

-----------

##### <a name="QuadraticStochNormClip"></a>	Quadratic Stoch Norm Clipping
about this clipping methods  

-----------

##### <a name="LayerWiseClip"></a>	Layer Wise Clipping
about this clipping methods  

-----------

##### <a name="CoordWiseClip"></a>	Coordinate Wise Clipping
about this clipping methods  

-----------

##### <a name="AutoClip"></a>	Auto Clipping
about this clipping methods  

-----------

##### <a name="LinearStochAutoClip"></a>	Linear Stoch Auto Clipping
about this clipping methods  

-----------

##### <a name="QuadraticStochAutoClip"></a>	Quadratic Stoch Auto Clipping
about this clipping methods  

-----------
