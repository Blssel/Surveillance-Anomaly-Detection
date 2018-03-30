# coding:utf-8
import numpy as np

thread=0.6
file_name='test_score_2775.txt'

with open(file_name,'r') as fout:
  score_list=fout.read().split('\n')
score_list.remove('')

for i in range(len(score_list)):
  score_list[i]=score_list[i].split(' ')

for j in range(len(score_list)):
  score_list[j][2]=float(score_list[j][2][1:-1])


# 按照第二项排序 由大到小
score_list.sort(lambda x,y:cmp(y[2],x[2]))

# 将所有数归一化为0-1
min_num=score_list[-1][2]
gap=score_list[0][2]-score_list[-1][2]
for i in range(len(score_list)):
  score_list[i][2]=(score_list[i][2]-min_num)/gap


for i in range(len(score_list)):
  print score_list[i]

# 以0.6作为分界点 分割成anomaly 和 normal
anomaly=[]
normal=[]
for i in range(len(score_list)):
  if score_list[i][2]>thread:
    anomaly.append(score_list[i])
  else:
    normal.append(score_list[i])

# 准确率
num_right1=0
num_right2=0
for i in range(len(anomaly)):
  if anomaly[i][0].startswith('Normal'):
    continue
  else:
    num_right1+=1

for i in range(len(normal)):
  if normal[i][0].startswith('Normal'):
    num_right2+=1
  else:
    continue
precision=(num_right1+num_right2)/290.0

# lou报率
loubaolv=float(len(normal)-num_right2)/140
 
# 误报率
wubaolv=float(len(anomaly)-num_right1)/len(anomaly)

#真阳率
true=float(num_right1)/140

#假阳率
false=float(len(anomaly)-num_right1)/150
print '正确率%f'%precision
print '漏报率%f'%loubaolv
print '误报率%f'%wubaolv
print '真阳率%f'%true
print '假阳率%f'%false
