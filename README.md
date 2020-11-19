# Quantization_EfficientNet
Start to learn the quantization.  Float32->Int8
Thanks to @qubvel's EfficientNet.
I use Tensorflow Quantization API to quantize the model.
The difficult thing is that how to figure out which line can be quaztized.
You have to distinguish tf and tf.keras. Otherwise, you will meet a variety of disgusting errors.


# Acknowledgements
Thanks to qubvel and tensorflow.
@qubvel's EfficientNet     : https://github.com/qubvel/efficientnet
Quantization aware training : https://www.tensorflow.org/model_optimization/guide/quantization/training




