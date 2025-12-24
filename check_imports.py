try:
    import tensorflow
    print("TensorFlow imported")
except ImportError as e:
    print(f"TensorFlow error: {e}")

try:
    import numpy
    print("Numpy imported")
except ImportError as e:
    print(f"Numpy error: {e}")
