# coding:utf-8
import os
import cv2
import numpy as np
import math
import tensorflow as tf


TRAIN_SPLIT_PATH='/share/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt'
TEST_SPLIT_PATH='/share/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt'
VID_PATH_BASE='/share/dataset/UCF_Crimes/Videos'
NUM_SHARDS=1500
OUTPUT_DIR='/home/yzy/dataset/ucf_crimes_tfrecord'
def get_vid_list(split_path):
  with open(split_path) as fout:
    return fout.read().split('\n')
    
# 将视频归一化为0-1
def norm(video):
  return np.float32(video)/255

def cont(tmp_list):
  v=tmp_list[0]
  for i in range(1,len(tmp_list)):
    v=np.concatenate((v, tmp_list[i]))
  return v

# 读取视频，归一化处理
def read_vid(item):
  vid_full_path=os.path.join(VID_PATH_BASE,item)
  video_name=item.split('/')[1]
  action_name=item.split('/')[0]
  is_normal= int(video_name.split('_')[0]=='Normal')
  # 读取视频，获取帧数，height width等参数,
  cap=cv2.VideoCapture(vid_full_path)# 选择用opencv来读取
  num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print num_frames
  height,width=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))# 高 宽
  
  #tmp_list=[]
  #tmp_cont=[]
  #num=0
  #while True:
  #  num+=1
  #  print num
  #  ret, frame=cap.read()
  #  if not ret: #如果没了，则整合tmp_list
  #    if len(tmp_list)==0:
  #      break
  #    else:
  #      tmp_cont.append(cont(tmp_list))
  #      break
  #  tmp_list.append(frame)
  #  if num==300:
  #    tmp_cont.append(cont(tmp_list))
  #    num=0
  #    tmp_list=[]
  #video=tmp_cont[0]
  #for i in range(1,len(tmp_cont)):
  #  video=np.concatenate((video,tmp_cont[i]))
  #  print '#########'
  
  ret,frame=cap.read()
  video=frame
  num=1
  # 为了提高效率，减少concatenate的负荷，设置每500次暂存一下,所以该部分注释掉 不用
  while(True):
    ret,frame=cap.read() #读取下一帧
    if not ret:
      break
    video=np.concatenate((video,frame))# 拼合
    num+=1
    print num
    if num>=300:
      break

  # reshape  并 归一化处理
  #video=video.reshape(num_frames,height,width,3)
  video=video.reshape(300,height,width,3)
  print 'reshape 完毕'
  print type(video)
  print video.shape
  video=norm(video) #归一化(变成float32类型了)
  return video_name,video,num_frames,height,width,is_normal,action_name

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
def int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
def trans_to_tfexample(data):
  video_name=data[0]
  video=data[1]
  num_frames=data[2]
  height=data[3]
  width=data[4]
  is_normal=data[5]
  action_name=data[6]
  # 将数据转换成字符串
  video=video.tostring()
  return tf.train.Example(features=tf.train.Features(feature={
                          'video/video_name':bytes_feature(video),
                          'video/encoded':bytes_feature(video),
                          'video/num_frames':int64_feature(num_frames) ,
                          'video/height':int64_feature(height),
                          'video/width':int64_feature(width),
                          'video/class/is_normal':int64_feature(is_normal) ,
                          'video/class/action_name':bytes_feature(action_name) }))


def _convert_to_tfrecord(data,output_dir):
  video_name=data[0]
  video=data[1]
  num_frames=data[2]
  height=data[3]
  width=data[4]
  is_normal=data[5]
  action_name=data[6]
  example=trans_to_tfexample(data)
  print 'ddddddddddddddddddddddddddddddddddddddd'
  return example

def main():
  split=['train','test']
  train_vid_list=get_vid_list(TRAIN_SPLIT_PATH)
  test_vid_list=get_vid_list(TEST_SPLIT_PATH)
  
  num_per_shard=int(math.ceil(len(train_vid_list)/float(NUM_SHARDS)))
  for shard_id in range(0,NUM_SHARDS):
    # 计算该轮的起始和终止视频
    start_index=shard_id*num_per_shard
    end_index=min(start_index+num_per_shard, len(train_vid_list)) # 采用左闭右开方式
    output_filename=os.path.join(OUTPUT_DIR, 'train_'+str(shard_id))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      # 读取每一个vid  
      for item in train_vid_list[start_index:end_index]:
        vid_full_path=os.path.join(VID_PATH_BASE,item)
        video_name,video,num_frames,height,width,is_normal,action_name=read_vid(item)
        print '至此  该视频读完了，准备转成example'
        example=_convert_to_tfrecord((video_name,video,num_frames,height,width,is_normal,action_name),OUTPUT_DIR)
        print '至此  转成example结束，  准备写入'
        tfrecord_writer.write(example.SerializeToString())
        print '完成1'

if __name__=='__main__':
  main()
