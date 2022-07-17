# TorchClippedOptimizers

`torch-clip` a library to improve optimization methods by clipping off heavy-tailed gradient. This makes it possible to increase the accuracy and speed of convergence during the training of neural networks on a specific number of tasks
------------
## our clipping methods

+ [Linear Stoch Norm Clipping](#LinearStochNormClip);  
+ [Quadratic Stoch Norm Clipping](#QuadraticStochNormClip);  
+ [Layer Wise Clipping](#LayerWiseClip);  
+ [Coordinate Wise Clipping](#CoordWiseClip);  
+ [Auto Clipping](#AutoClip);  
+ [Linear Stoch Auto Clipping](#LinearStochAutoClip);  
+ [Quadratic Stoch Auto Clipping](#QuadraticStochAutoClip).
