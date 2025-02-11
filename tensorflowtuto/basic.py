import os
import tensorflow as tf

# TensorFlow 설치 위치 확인
tf_path = os.path.dirname(tf.__file__)
print("TensorFlow 설치 경로:", tf_path)

# keras 폴더 확인
keras_path = os.path.join(tf_path, "keras")
print("Keras 모듈 존재 여부:", os.path.exists(keras_path))

# # 0D (스칼라)
# scalar = tf.constant(5)
# print(scalar)
#
# # 1D (벡터)
# vector = tf.constant([1, 2, 3])
# print(vector)
#
# # 2D (행렬)
# matrix = tf.constant([[1, 2], [3, 4]])
# print(matrix)
#
# # 3D (텐서)
# tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(tensor)
#
# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[5, 6], [7, 8]])
#
# print(tf.add(a, b))   # 덧셈
# print(tf.subtract(a, b))  # 뺄셈
# print(tf.multiply(a, b))  # 곱셈
# print(tf.divide(a, b))  # 나눗셈
#
# print(tf.matmul(a, b))  # 행렬 곱
#
# print(tf.config.list_physical_devices("GPU"))
#
# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[5, 6], [7, 8]])
#
# print(tf.add(a, b))   # 덧셈
# print(tf.subtract(a, b))  # 뺄셈
# print(tf.multiply(a, b))  # 곱셈
# print(tf.divide(a, b))  # 나눗셈