# TorchClippedOptimizers


`torch-clip` a library to improve optimization methods by clipping off heavy-tailed gradient. This makes it possible to increase the accuracy and speed of convergence during the training of neural networks on a specific number of tasks.

**Example of the distribution of gradient lengths with heavy tails:**

![This is an image](readme_images/heavy_tail.jpg)
------------

### Installation
you can install our library using pip:  
`pip install torch-clip`  
```requirements.txt
numpy~=1.20.0
torch~=1.11.0+cu113
matplotlib~=3.4.3
tqdm~=4.62.3
```

### What do you need us for?
In the last few years, for various neural network training models (for example, BERT + CoLA), it has been found that in the case of "large stochastic gradients", it is advantageous to use special clipping (clipping/normalization) of the batched gradient. Since all modern machine learning, one way or another, ultimately boils down to stochastic optimization problems, the question of exactly how to "clip" large values of batched gradients plays a key role in the development of effective numerical training methods for a large class of models. This repository implements optimizers for the pytorch library with different clipping methods.


### Our clipping methods

+ [Norm Clipping](#NormClip)
+ [Linear Random Norm Clipping](#LinearRandNormClip);  
+ [Quadratic Random Norm Clipping](#QuadraticRandNormClip);  
+ [Layer Wise Clipping](#LayerWiseClip);  
+ [Coordinate Wise Clipping](#CoordWiseClip);  
+ [Auto Clipping](#AutoClip);  
+ [Linear Random Auto Clipping](#LinearRandAutoClip);  
+ [Quadratic Random Auto Clipping](#QuadraticRandAutoClip).


### Comparison on different tasks
We conducted a study to study the quality of our clipping methods on a number of tasks: image classification, semantic segmentation, text classification and graph-node classification.  
#### Image Classification on ImageNet dataset and Resnet18 model:  
![This is an image](readme_images/image-classification.png) 
#### Semantic Segmentation on PascalVOC dataset and Unet model:  
![This is an image](readme_images/semnatic-segmentation.png) 
#### Text Classification on CoLA dataset and Bert model:  
![This is an image](readme_images/text-classification.jpg) 
#### Graph-Node classifcation on Reddit node dataset and custom GraphConv model:  
![This is an image](readme_images/graph-node-classification.jpg) 

<br>
<br>

#### <a name="NormClip"></a> Norm Clipping
Norm-clipping is a basic clipping method that uses a constant to clip gradient.
$$\alpha_{norm} = {\frac{\eta}{||\nabla f(x^k, \xi^k)||_2}}$$

-----------
<br>
<br>

#### <a name="LinearRandNormClip"></a> Linear Random Norm Clipping
LinearRandNormClip is a norm-clipping method using randomization when clipping gradient, which helps to shift the mathematical expectation less.
$$P(\text{clip})=\beta^{\alpha_{\text{norm}}}, \text{where}\ 0<\beta<1 \text{ and}\ \alpha = \alpha_{\text{norm}}$$

-----------
<br>

#### <a name="QuadraticRandNormClip"></a>	Quadratic Random Norm Clipping
QuadraticRandNormClip is a norm-clipping method using randomization when clipping gradient and increasing the probability of clipping by squaring.
$$P(\text{clip})=\beta^{\alpha_{\text{norm}}^2},\text{where}\ 0<\beta<1 \text{ and}\ \alpha = \alpha_{\text{norm}}$$

-----------
<br>

#### <a name="LayerWiseClip"></a>	Layer Wise Clipping
LayerWiseClip is a constant clipping method that clips gradients for each layer of the model separately

$$\alpha_{\text{layer}} = \frac{\eta}{||\nabla_{w_{l}} f(x^k,\xi^k)||_2}, \text{where}\ w_l - \text{ weights of current layer in neural network}\ $$

-----------
<br>

#### <a name="CoordWiseClip"></a>	Coordinate Wise Clipping
CoordWiseClip is a constant clipping method that clips gradients for each model parameters of the model separately (because of this, the direction of the gradient vector may change)
$$\alpha_w = \frac{\eta}{|\frac{\partial f}{\partial w}(x^k, \xi^k)|},  w - \text{current model's parameter}\$$

-----------
<br>

#### <a name="AutoClip"></a>	Auto Clipping
AutoClip is a clipping method that automatically selects the pth percentile in the gradient length distribution and uses it as a parameter for clipping.
$$\alpha_{\text{auto}} = \frac{\eta(p)}{||\nabla f(x^k,\xi^k)||_2}, \text{where}\  0 < p \leq 1 \text{ and}\ \eta(p) - \text{p-th percentile}\$$

-----------
<br>

#### <a name="LinearRandAutoClip"></a>	Linear Random Auto Clipping
LinearRandAutoClip is an auto-clipping method, using randomization when clipping gradient, which helps to shift the mathematical expectation less.
$$P(\text{clip})=\beta^{\alpha_{\text{auto}}}, \text{where}\ 0<\beta<1 \text{ and}\ \alpha = \alpha_{\text{auto}} $$

-----------
<br>

#### <a name="QuadraticRandAutoClip"></a>	Quadratic Random Auto Clipping
QuadraticRandAutoClip is an automatic clipping method that uses randomization when clipping gradient and squaring the clipping probability.
$$P(\text{clip})=\beta^{\alpha_{\text{auto}}^2}, \text{where}\ 0<\beta<1 \text{ and}\ \alpha = \alpha_{\text{auto}}$$

-----------
<br>

### Use example  
You can use our optimizers as well as all the standard optimizers from the pytorch library  
```python
from torch_clip.optimizers import  ClippedSGD
optimizer = ClippedSGD(model.parameters(), lr=5e-2, momentum=0.9, clipping_type="layer_wise", clipping_level=1)

loss = my_loss_function
for epoch in range(EPOCHS):
    for i, data in enumerate(train_loader, 0):
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

<br>
<br>

### Use example (with restarts)

```python
from torch_clip.optimizers import ClippedSGD
from torch_clip.restarter import Restarter
from torch_clip.optimizers_collector import OptimizerProperties, ModelProperties, RestartProperties

loss = my_loss_function
model = my_model_object

optimizer_props = OptimizerProperties(ClippedSGD, lr=5e-2, momentum=0.9, 
                                      clipping_type="layer_wise", clipping_level=1)
restarter = Restarter(optimizer_properties=optimizer_props, first_restart_steps_cnt=50,
                      restart_coeff=1.25, max_steps_cnt=2000)
optimizer = optimizer_props.optimizer_class(model.parameters(), **optimizer_props.optimizer_kwargs)

for epoch in range(EPOCHS):
    for i, data in enumerate(train_loader, 0):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        restarter.add_coords(model.parameters())
        optimizers = restarter.make_restart(net, optimizer)
```
