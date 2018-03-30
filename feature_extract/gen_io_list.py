#coding:utf-8
import os
import cv2

#video_path='/extra_disk/dataset/UCF_Crimes/Videos'
#output_path='/extra_disk/dataset/UCF_Crimes_C3D_features/Videos'
video_path='/share/dataset/UCF_Crimes/Videos'
output_path='UCF_Crimes_C3D_features/Videos'
input_file='./ucfcrime_input_list_video.txt'
output_file='./ucfcrime_output_list_video_prefix.txt'

f1=open(input_file,'w')
f2=open(output_file,'w')

for cur_location,dir_names,file_names in os.walk(video_path):
  if file_names==None:
    continue
  else:
    # 对每一个视频
    action_name=cur_location.split('/')[-1]
    for vid in file_names:
      # 先为该视频创建对应输出文件夹
      outdir=os.path.join(output_path,action_name,vid.split('.')[0])
      if not os.path.exists(outdir):
        os.makedirs(outdir)
      # 读取视频，并获取总帧数
      cap=cv2.VideoCapture(os.path.join(cur_location,vid))
      num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      # 输入文件三元组
      vid_fullpath=os.path.join('/extra_disk/dataset/UCF_Crimes/Videos/frames',action_name,vid.split('.')[0])
      for frame_index in range(num_frames):
        if frame_index%32==0 and (frame_index+15)<num_frames:
          start_frame=frame_index
          f1.write(vid_fullpath+' '+str(start_frame+1)+' '+'0'+'\n')
          # 输出文件一元组
          f2.write(os.path.join(outdir,'%06d'%frame_index)+'\n')
    
f1.close()
f2.close()
