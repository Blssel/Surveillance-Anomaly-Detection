import tensorflow as tf
import numpy as np

#with tf.Graph().as_default():
a=tf.constant([1,2,3])
length=a.shape.as_list()
print length
print 'length=%d'%length[0]


with tf.Session() as sess:
  print '1'
  print(sess.run(a))

b=tf.constant([4,5,6])
c=tf.add(a,b)

with tf.Session() as sess:
  print '2'
  print sess.run(a)
  print sess.run(c)
