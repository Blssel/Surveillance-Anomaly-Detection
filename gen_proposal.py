#coding:utf-8
import os
import glob
import tensorflow as tf
import random
import r
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#flags = tf.app.flags
gpu_num = 2
#flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('batch_size', 10, 'Batch size.')
#FLAGS = flags.FLAGS
batch_size=30
max_steps=5000
test_split_path='/extra_disk/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt'
#train_split_path='/share/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt'
model_save_dir='models/ucf-crimes'
lamb1=8e-5
lamb2=8e-5

def fc_inference(_X, _dropout, batch_size, _weights, _biases):
  fc_7 = tf.matmul(_X, _weights['w1']) + _biases['b1']
  fc_7 = tf.nn.relu(fc_7, name='fc7') # Relu activation
  fc_7 = tf.nn.dropout(fc_7, _dropout)  

  fc_8 = tf.matmul(fc_7, _weights['w2']) + _biases['b2']
  fc_8 = tf.nn.sigmoid(fc_8, name='fc8') # Relu activation
  fc_8 = tf.nn.dropout(fc_8, _dropout)  

  out=tf.matmul(fc_8, _weights['w3'])

  return out

def placeholder_inputs(batch_size):
  feature_placeholder_ano = tf.placeholder(tf.float32, shape=(None,4096))
  ano_segment_ids= tf.placeholder(tf.int32, shape=(None)) #每个位置存放该视频的seg数量
  feature_placeholder_no = tf.placeholder(tf.float32, shape=(None,4096))
  no_segment_ids= tf.placeholder(tf.int32, shape=(None)) #每个位置存放该视频的seg数量
  return feature_placeholder_ano, ano_segment_ids, feature_placeholder_no, no_segment_ids

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

def read_one(item):
  full_path=os.path.join('/home/zy_17/workspace/Surveillance-Anomaly-Detection/feature_extract/UCF_Crimes_C3D_features/Videos', item.split(.)[0])
  
  feature_placeholder=[]
  segment_ids=[] 
 
  feature_list=glob.glob(full_path)
  if len(feature_list)==0:
    #header, data =  r.read_binary_fc('/home/yzy/000000.fc6-1')
    header, data =  r.read_binary_fc('/home/zy_17/000000.fc6-1')
    feature_placeholder.append(data)
    segment_ids.append(0)
  else:
    for ind,seg in enumerate(feature_list):
      # 读取该条数据
      header, data =  r.read_binary_fc(os.path.join(full_path, seg.split('/')[-1]))
      # 把data加入
      feature_placeholder.append(data)
      segment_ids.append(0)

  feature_placeholder =np.array(feature_placeholder)
  segment_ids =np.array(segment_ids)
  return feature_placeholder,segment_ids
  


def run_testing():
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)

  global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(181),trainable=False)
  feature_placeholder_ano, ano_segment_ids, feature_placeholder_no, no_segment_ids = placeholder_inputs(batch_size)

  with tf.variable_scope('fc') as var_scope:
    weights = {
            'w1': _variable_with_weight_decay('w1', [4096, 512], 0.0005),
            'w2': _variable_with_weight_decay('w2', [512,32], 0.0005),
            'w3': _variable_with_weight_decay('w3', [32,1], 0.0005),
            }
    biases = {
            'b1': _variable_with_weight_decay('b1', [512], 0.000),
            'b2': _variable_with_weight_decay('b2', [32], 0.000),
            }
  varlist = list( set(weights.values()) )
  # 前传
  ano_out = fc_inference(feature_placeholder_ano, 0.6, batch_size, weights, biases)
  no_out = fc_inference(feature_placeholder_no, 0.6, batch_size, weights, biases)

'''    
  # 计算平滑约束loss
  #with tf.Session() as sess:
  #  len_batch=tf.shape(ano_segment_ids).eval()[0]
  try:
    difference_list=[tf.subtract(ano_out[i+1],ano_out[i]) for i in range(0,1000)]
  except:
    print '越界'
    pass
  print '已跳出'
  smooth_loss=tf.reduce_mean(tf.square(tf.stack(difference_list)))
'''
  '''
  difference_list=[]
  i=0
  while(True):
    try:
      print 'a'
      difference_list.append(tf.subtract(ano_out[i+1],ano_out[i]))
      i+=1
    except:
      print 'bbbb'
      break
  smooth_loss=tf.reduce_mean(tf.square(tf.concat(difference_list)))
  '''
  # 计算稀疏约束loss
  sparse_loss=tf.reduce_sum(ano_out)
  # 选出top1
  top1max_ano_out = tf.segment_max(ano_out,ano_segment_ids)
  top1max_ano_out_index = tf.where(tf.equal(ano_out,max_ano_out))
  # 选出top2
  ano_out[top1max_ano_out_index]=tf.constant([-float(inf)]) # 先给上一个最大位置上的得分置负无穷小
  top2max_ano_out = tf.segment_max(ano_out,ano_segment_ids)
  top2max_ano_out_index = tf.where(tf.equal(ano_out,max_ano_out))
  # 选出top3
  ano_out[top2max_ano_out_index]=tf.constant([-float(inf)])
  top3max_ano_out = tf.segment_max(ano_out,ano_segment_ids)
  top3max_ano_out_index = tf.where(tf.equal(ano_out,max_ano_out))
  
  top1max_ano_out = tf.squeeze(top1max_ano_out_index)
  top2max_ano_out = tf.squeeze(top2max_ano_out_index)
  top3max_ano_out = tf.squeeze(top3max_ano_out_index)

  score1= top1max_ano_out
  score2= top2max_ano_out
  score3= top3max_ano_out

'''
  print ano_out
  no_out = tf.segment_max(no_out,no_segment_ids)
  print no_out
  # 计算hinge_loss
  hinge_loss=tf.reduce_mean(tf.maximum(
                                0.0,
                                tf.subtract(1.0,tf.subtract(ano_out,no_out))
                                )
                            )
  # 计算正则化loss
  regularizer=tf.contrib.layers.l2_regularizer(scale=8e-5)
  regulation=tf.contrib.layers.apply_regularization(regularizer, weights_list=varlist)
  # loss
  loss = hinge_loss+ lamb1*smooth_loss+ lamb2*sparse_loss+ regulation
  #loss = hinge_loss+ lamb2*sparse_loss+ regulation


  # 优化      
  learning_rate = 0.001
  train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step,var_list=varlist)
  saver = tf.train.Saver(var_list=varlist,reshape=True)
'''   
  
  saver2 = tf.train.Saver(var_list=varlist,reshape=True,max_to_keep=100)



  with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    saver2.restore(sess,'./model/bk.ckpt-800')

    sess.graph.finalize()
    # 读取视频list########################
    with open(test_split_path,'r') as fout:
      test_list=fout.read().split('\n')
    test_list.remove('')
     
    # 对每一个视频，获得他的得分 
    for item in test_list:
      vid=item.split('/')[-1]
      propty='normal' if item.split('/')==‘Testing_Normal_Videos_Anomaly’ 'anomaly'
      # 读取该视频
      r_feature_placeholder, r_segment_ids=read_one(item)
      score=sess.run([score],feed_dict={feature_placeholder_no: r_feature_placeholder,
                                          ano_segment_ids:r_segment_ids})
      with open('test_score.txt','w') as fout:
        fout.write(vid+' '+propty+' '+str(score))
      

def main():
  run_testing() 

if __name__=='__main__':
  main()
