#coding:utf-8
import os
import glob
import tensorflow as tf
import random
import r
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#flags = tf.app.flags
gpu_num = 2
#flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('batch_size', 10, 'Batch size.')
#FLAGS = flags.FLAGS
batch_size=30
max_steps=5000
train_split_path='/data1/zhiyuyin/dataset/ucf_crimes/Anomaly_Detection_splits/Train.txt'
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
  feature_placeholder_ano = tf.placeholder(tf.float32, shape=(None,1024))
  ano_segment_ids= tf.placeholder(tf.int32, shape=(None)) #每个位置存放该视频的seg数量
  feature_placeholder_no = tf.placeholder(tf.float32, shape=(None,1024))
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


def next_batch(batch_size,ano_list,no_list):
  feature_placeholder_ano=[]
  feature_placeholder_no=[]
  ano_segment_ids=[]
  ano_index=-1
  no_segment_ids=[]
  no_index=-1
  for i in range(batch_size):
    #print '读了一个'
    ano_index+=1
    no_index+=1
    # 在两个list中各随机选取一个video，把每个segments加入集合
    rand_ano=random.randint(0,len(ano_list)-1)
    rand_no=random.randint(0,len(no_list)-1)
    #ano_feature_path=os.path.join('/home/zy_17/workspace/C3D/C3D-v1.0/UCF_Crimes_C3D_features/Videos' , ano_list[rand_ano].split('.')[0])
    #no_feature_path=os.path.join('/home/zy_17/workspace/C3D/C3D-v1.0/UCF_Crimes_C3D_features/Videos' , no_list[rand_no].split('.')[0] )
    ano_feature_path=os.path.join('/data1/zhiyuyin/experiments/YZYN-Anomaly-Detection/features_16_segm' , os.path.basename(ano_list[rand_ano]).split('.')[0])
    no_feature_path=os.path.join('/data1/zhiyuyin/experiments/YZYN-Anomaly-Detection/features_16_segm' , os.path.basename(no_list[rand_no]).split('.')[0] )
    #ano_feature_path=os.path.join('/home/yzy/Videos' , ano_list[rand_ano].split('.')[0])
    #no_feature_path=os.path.join('/home/yzy/Videos' , no_list[rand_no].split('.')[0] )
    # 读取所有数据，放入list，然后逐个读取存入feature_placeholder_ano
    ano_feature_list=glob.glob(os.path.join(ano_feature_path,'*'))
    no_feature_list=glob.glob(os.path.join(no_feature_path,'*'))
    if len(ano_feature_list)==0:
      #header, data =  r.read_binary_fc('/home/yzy/000000.fc6-1')
      header, data =  r.read_binary_fc('/home/zy_17/000000.fc6-1')
      feature_placeholder_ano.append(data)
      ano_segment_ids.append(ano_index)
    else:
      for ind,seg_ano in enumerate(ano_feature_list):
        # 读取该条数据
        #print seg_ano
        # header, data =  r.read_binary_fc(os.path.join(ano_feature_path, seg_ano.split('/')[-1]))
        data = np.load(seg_ano)
        # 把data加入
        feature_placeholder_ano.append(data)
        # 做一个标记
        ano_segment_ids.append(ano_index)
    if len(no_feature_list)==0:
      #header, data =  r.read_binary_fc('/home/yzy/000000.fc6-1')
      header, data =  r.read_binary_fc('/home/zy_17/000000.fc6-1')
      feature_placeholder_no.append(data)
      no_segment_ids.append(no_index)
    else:
      for ind,seg_no in enumerate(no_feature_list):
        # 读取该条数据
        #print seg_no
        #header, data =  r.read_binary_fc(os.path.join(no_feature_path, seg_no.split('/')[-1]))
        data = np.load(seg_no)
        # 把data加入
        feature_placeholder_no.append(data)
        # 做一个标记
        no_segment_ids.append(no_index)
  #print 'batch 读取完毕'
  feature_placeholder_ano=np.array(feature_placeholder_ano)
  ano_segment_ids=np.array(ano_segment_ids)
  feature_placeholder_no=np.array(feature_placeholder_no)
  no_segment_ids=np.array(no_segment_ids)
  #print feature_placeholder_ano.shape
  return feature_placeholder_ano, ano_segment_ids, feature_placeholder_no, no_segment_ids
    

def run_training():
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)

  global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(2654),trainable=False)
  feature_placeholder_ano, ano_segment_ids, feature_placeholder_no, no_segment_ids = placeholder_inputs(batch_size)

  with tf.variable_scope('fc') as var_scope:
    weights = {
            'w1': _variable_with_weight_decay('w1', [1024, 512], 0.0005),
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
  print '@@@@@@@@@@@@@@@@@@@'
  print ano_out.shape.as_list()
  print '@@@@@@@@@@@@@@@@@@@'
  no_out = fc_inference(feature_placeholder_no, 0.6, batch_size, weights, biases)
    
  # 计算平滑约束loss
  #with tf.Session() as sess:
  #  len_batch=tf.shape(ano_segment_ids).eval()[0]
  try:
    difference_list=[tf.subtract(ano_out[i+1],ano_out[i]) for i in range(0,batch_size*32-1)]
  except:
    print '越界'
    pass
  print '已跳出'
  smooth_loss=tf.reduce_mean(tf.square(tf.stack(difference_list)))
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
  # 精简结果为两个batch_zie*1
  ano_out = tf.segment_max(ano_out,ano_segment_ids)
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
   
  
  saver2 = tf.train.Saver(var_list=varlist,reshape=True,max_to_keep=100)



  with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #sess.run(global_step.initializer)
    #saver2.restore(sess,'./finetune_model/')
    #导入pre_trained模型变量
    #saver.restore(sess,MODEL_NAME)
    #saver2.restore(sess,'./model/bk.ckpt-2654')

    sess.graph.finalize()
    # 读入训练列表
    with open(train_split_path,'r') as fout:
      train_list=fout.read().split('\n')
    ano_list=[]
    no_list=[]
    for item in train_list:
      if item.split('/')[0]=='Training_Normal_Videos_Anomaly':
        no_list.append(item)
      else:
        ano_list.append(item)
    #ano_list.remove('')
    #print ano_list

    for i in range(max_steps):
      r_feature_placeholder_ano, r_ano_segment_ids, r_feature_placeholder_no, r_no_segment_ids = next_batch(batch_size,ano_list,no_list)#!!!!在这里读取参数
      #print '至少batch读入成功了'
      _,loss_value, step = sess.run([train_step,loss, global_step], feed_dict={feature_placeholder_ano: r_feature_placeholder_ano, 
                                                                               ano_segment_ids:r_ano_segment_ids,
                                                                               feature_placeholder_no:r_feature_placeholder_no,
                                                                               no_segment_ids:r_no_segment_ids   })
      #print 'hinge_loss is %f'%hinge_loss
      #print 'sparse_loss is %f'%sparse_loss
      if i % 5 == 0:
        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
      if i % 20 == 0:
        saver.save(sess, './model/bk.ckpt', global_step=global_step)


def main():
  run_training() 

if __name__=='__main__':
  main()
