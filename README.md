# Quantization_EfficientNet
Start to learn the quantization.  Float32->Int8  
Thanks to @qubvel's EfficientNet.  
I use Tensorflow Quantization API to quantize the model.  
The difficult thing is that how to figure out which line can be quaztized.  
You have to distinguish tf and tf.keras. Otherwise, you will meet a variety of disgusting errors.      

My Top1 accuracy average is 76%. Top 5 is 96%  
Training time is 35 mins.  

# Google Colaboratory 
If you don't have a sufficient device to run machine learning, I recommend Google Colaboratory.  
Google Colaboratory provides you Tesla K80 GPU and TPU.  



# Acknowledgements
Thanks to @qubvel, tensorflow and Google Colaboratory.  
@qubvel's EfficientNet      : https://github.com/qubvel/efficientnet    
Quantization aware training : https://www.tensorflow.org/model_optimization/guide/quantization/training    
Google Colaboratory         : https://colab.research.google.com/notebooks/intro.ipynb  



