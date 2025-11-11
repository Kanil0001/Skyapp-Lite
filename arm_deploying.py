import tensorflow.lite as tflite
interpreter = tflite.Interpreter(model_path="skymind.tflite")
interpreter.allocate_tensors()
