import tensorflow as tf
# import tf.keras.backend as K
import numpy as np

from keras import backend as K


a = np.random.randn(5,5)
ones = np.ones((2,3))
b = K.variable(ones,dtype='float32')



a = K.variable(a)
a = K.cast( K.greater(a,0.8) , 'float32' )
sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(K.mean(b,axis=1)))
# print(sess.run(a),'\n')
# print(sess.run(a*b))
sess.close()