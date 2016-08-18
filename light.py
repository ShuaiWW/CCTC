
import numpy as np

import matplotlib.pyplot as plt
from cvxpy import *

def generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5,grind=4):
  # user_num is number of users on each ZigBee channel
  # packet_len is the length of each Zigbee channel
  # how_long is how long time this simulation runs
  RSSI=[]
  intervals=[53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149]

  #intervals=[11,13,17,19,23,31,37,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163];
  interval1=[17,31,47,89,109,101,103,107]
  interval2=[19,37,53,97,139,127,131,137]
  interval3=[23,43,59,113,149,151,157,163]
  #starts=np.random.randint(interval[0],size=14)
  #start=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  data=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
  #data=np.random.randint(5,size=16)
  RSSI_channel=[]
  for channel in range(len(user_num)):
    num=user_num[channel]
    temp=np.zeros(int(1000*how_long*grind*packet_len[channel]))
    already=[]
    for i in range(num):
      signal_energy=np.random.random_sample()*0.01
      #start=np.random.randint(100*grind*packet_len[channel]);
      overlap=True
      print channel,i,already
      while overlap:
        start=np.random.randint(100*grind*packet_len[channel]);
        overlap=False
        
        for ss in already:
          overlap= overlap or ( start>=ss-packet_len[channel]*grind and start <= ss+packet_len[channel])
        print overlap, start,already
      already.append(start)



      for j in range(int(1000*how_long*grind*packet_len[channel])):
        if j>=start:
          if (j-start)%(intervals[i]*grind*packet_len[channel])==0:
            for k in range(grind*packet_len[channel]):
              if j+k<int(1000*how_long*grind*packet_len[channel]) and temp[j+k]==0:
                temp[j+k]=signal_energy+0.01;
          if (j-start-data[i]*grind*packet_len[channel])%(intervals[i]*grind*packet_len[channel])==0:
            for k in range(grind*packet_len[channel]):
              if j+k<int(1000*how_long*grind*packet_len[channel]) and temp[j+k]==0:
                temp[j+k]=signal_energy+0.01;
    RSSI_channel.append(list(temp))
  for time in range(int(1000*how_long*grind*packet_len[0])):
    temp=[];
    for i in range(len(user_num)):
      temp.append(RSSI_channel[i][time])
    RSSI.append(temp)
  return RSSI

def shrink(signal,granularity,packet_len,output_granularity):
  result=[]
  index=0
  while index<len(signal):
    temp=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    print index
    for i in range(int(packet_len*output_granularity/granularity)):
      for ii in range(14):
        if signal[index+i][ii]>=0.0001:
          temp[ii]=1
    index+=int(packet_len*output_granularity/granularity)
    result.append(temp)
  return result

## old version
def switch_sense(start,signal,magnification,granularity,delay,output_granularity):
  # this function would sample on "Signal" with "delay". Each sample would be gotten by sensing energy and switching channels
  # The metric of granularity and delay is millisecond
  index=start
  mask=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
  energy=[]
  sample_matrix=[]
  column=14*len(signal)*granularity/(magnification*output_granularity);
  now=0
  sampled={}
  while now<len(signal)*granularity-delay:
    
    now=index*delay
    temp=np.zeros(column)
    channels=mask[index%11:index%11+4]
    sum=0
    index+=1
    output_index=int(now/(magnification*output_granularity))
    if sampled.get(output_index)==None:
      sampled[output_index]=1
    else:
      if sampled[output_index]<11:
        sampled[output_index]+=1
      else:
        continue

    for ZigBee_channel in channels: 
      sum+=signal[int(now/granularity)][ZigBee_channel]
      temp[14*int(now/(magnification*output_granularity))+ZigBee_channel]=1
    energy.append(sum)
    sample_matrix.append(temp)
    print now
    
  return np.matrix(sample_matrix),np.matrix(energy)



def switch_sense(sync_start,start,packet_len,signal,magnification,granularity,delay,output_granularity):
  # new version
  # this function would sample on "Signal" with "delay". Each sample would be gotten by sensing energy and switching channels
  # The metric of granularity and delay is millisecond
  index=start
  #packet_len=0.7
  mask=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
  energy=[]
  sample_matrix=[]
  new_matrix=[]
  column=14*len(signal)*granularity/(magnification*output_granularity);
  now=0
  sampled={}
  sync_index=sync_start
  sync_matrix=[]
  sync_count=0
  real_sync=[]
  while now<(len(signal)*granularity-delay+start*delay):
    
    now=index*delay
    temp=np.zeros(column)
    tep=np.zeros(14)
    channels=mask[index%11:index%11+4]
    sum=0
    index+=1
    output_index=int((now-start*delay)/(magnification*output_granularity))
    if sampled.get(output_index)==None:
      sampled[output_index]=1
    else:
      if sampled[output_index]<11:
        sampled[output_index]+=1
      else:
        continue

    for ZigBee_channel in channels: 
      sum+=signal[int((now-start*delay)/granularity)][ZigBee_channel]
      temp[14*int((now-start*delay)/(magnification*output_granularity))+ZigBee_channel]=1
      tep[ZigBee_channel]=1
    
    # construct sync matrix 
    if now-sync_index*packet_len>0 and now-sync_index*packet_len<delay+0.001:
      sync_temp=np.zeros(len(signal))
      sync_t=np.zeros(len(signal))
      sync_t[index]=1
      real_sync.append(sync_t)
      #print sync_t
      for i in range(sync_count):
        sync_temp[index-i]=1
      sync_matrix.append(sync_temp)

      sync_index+=1
      sync_count=1;
    else:
      sync_count+=1
      sync_temp=np.zeros(len(signal))
      sync_matrix.append(sync_temp)
      real_sync.append(sync_temp)

    print sync_count

    energy.append(sum)
    sample_matrix.append(temp)
    new_matrix.append(tep)
    print now,now-sync_index*packet_len,sync_index,now-sync_index*packet_len>0,now-sync_index*packet_len<=delay
    
  return np.matrix(sample_matrix),np.matrix(energy),np.matrix(new_matrix),np.matrix(sync_matrix),np.matrix(real_sync),index%11


def reform(channel_num,result):
  #vv=result.value
  temp=result.transpose().tolist()[0]
  res=[]
  for k in range(len(temp)/channel_num):
    tt=[]
    for i in range(channel_num):
      tt.append(temp[k*channel_num+i])
    res.append(tt)
  return res


def enlarge(ZigBee):
  result=[]
  for item in ZigBee:
    temp=[]
    for signal in item:
      temp.append(signal)
    result.append(temp)
    result.append(temp)
    result.append(temp)
  return result


def diff(z1,z2):
  import numpy as np
  error=0
  correct=0
  non_zero_correct=0
  all=0;
  raw_correct=0
  b_positive=0;

  for item in range(len(z1)):
    for ii in range(len(z1[item])):
      if z1[item][ii]<=0.0001:
        z1[item][ii]=0
      else:
        z1[item][ii]=1
  for item in range(len(z2)):
    for ii in range(len(z2[item])):
      if z2[item][ii]<=0.0001:
        z2[item][ii]=0
      else:
        z2[item][ii]=1
  for i in range(min(len(z1),len(z2))):
    for index in range(len(z1[i])):
      all+=1
      a=z1[i][index]
      b=z2[i][index]
      if b!=0:
        b_positive+=1
      if a!=0:
        raw_correct+=1
      if a!=0 and b!=0 :
        non_zero_correct+=1
      if a==b:
        correct+=1
  return all,raw_correct,correct,non_zero_correct,b_positive


def cal(u_num):
  res=[]
  for i in range(14):
    res.append(u_num)
  return res

def generate_sync(start,sample_num,length):
  result=[]
  real_sync=np.eye(length)
  for index in range(length):
    temp=np.zeros(length)
    if (index - start )% sample_num==0:
      sync_index=int(index/sample_num)
      for i in range(sample_num):
        temp[sync_index*sample_num+i]=1
    else:
      real_sync[index][index]=0
    result.append(temp)

  return np.matrix(result),real_sync
def generate_sync(start,sample_num,length):
  result=np.zeros((length,length))
  real_sync=np.zeros((length,length))
  for i in range(length):
    if (i-start)%sample_num==0:
      print i
      real_sync[i,i]=1
      if i+sample_num>length:
        break;
      for j in range(sample_num):
        result[i,i+j]=1

  return result,real_sync




def ext(delay,packet_len,signal_len):
  row_num=int(signal_len/(packet_len/delay))
  result=[]
  for i in range(row_num):
    temp=np.zeros(signal_len)
    for j in range(int(packet_len/delay)):
      temp[i*int(packet_len/delay)+j]=1
    result.append(temp)
  return np.matrix(result)

def weave(potential,error):
  result=[]
  result=potential[0][:]
  for i in range(1,len(error)):
    print i 
    if error[i]<=0.04:
      print error[i]
      for index in range(len(result)-len(potential[i]),len(result)):
        result[index]=potential[i][index-(len(result)-len(potential[i]))][:]
      result.append(potential[i][len(potential[i])-1][:])
    #else:

  return result


def average_signal(signal,window):
  average=[]
  variance=[]

  for ss in signal:
    temp_avg=[]
    temp_var=[]
    for i in range(len(ss)-window):
      t_avg=sum(ss[i:i+window])*1.0/window

      temp_avg.append(t_avg)
      t_var=0
      for item in ss[i:i+window]:
        t_var+=(item-t_avg)*(item-t_avg)
      t_var=t_var/window
      temp_var.append(t_var)
    average.append(temp_avg)
    variance.append(temp_var)
  return average,variance


def weave(potential,error):
  result=[]
  result=potential[0][:]
  temp_plus=0
  for i in range(1,len(error)):
    print i 
    if error[i]<=0.01:
      print error[i],len(potential[i])-1-temp_plus
      start=len(result)+temp_plus-len(potential[i])
      for index in range(start,len(result)):
        result[index]=potential[i][index-start][:]
      for index in range(len(potential[i])-1-temp_plus,len(potential[i])):
        print index
        result.append(potential[i][index][:])
      temp_plus=0
    else:
      temp_plus+=1
      if temp_plus>len(potential[i])-1:
        temp_plus-=1
        result.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  return result




def concatenate(all_signal,part_signal):
  # part_signal is a numpy matrix and all signal is a list I think.
  new=part_signal.transpose().tolist()
  for item in new:
    all_signal.append(item)
  return all_signal







def switch_sense(sync_start,start,packet_len,signal,magnification,granularity,delay,output_granularity):
  # new version
  # this function would sample on "Signal" with "delay". Each sample would be gotten by sensing energy and switching channels
  # The metric of granularity and delay is millisecond
  index=start
  #packet_len=0.7
  mask=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
  energy=[]
  sample_matrix=[]
  new_matrix=[]
  column=14*len(signal)*granularity/(magnification*output_granularity);
  now=0
  sampled={}
  sync_index=sync_start
  sync_matrix=[]
  sync_count=0
  real_sync=[]
  while now<(len(signal)*granularity-delay+start*delay):
    
    now=index*delay
    temp=np.zeros(column)
    tep=np.zeros(14)
    channels=mask[index%11:index%11+4]
    sum=0
    index+=1
    output_index=int((now-start*delay)/(magnification*output_granularity))
    if sampled.get(output_index)==None:
      sampled[output_index]=1
    else:
      if sampled[output_index]<11:
        sampled[output_index]+=1
      else:
        continue

    for ZigBee_channel in channels: 
      sum+=signal[int((now-start*delay)/granularity)][ZigBee_channel]
      temp[14*int((now-start*delay)/(magnification*output_granularity))+ZigBee_channel]=1
      tep[ZigBee_channel]=1
    
    # construct sync matrix 
    if now-sync_index*packet_len>0 and now-sync_index*packet_len<delay+0.001:
      sync_temp=np.zeros(len(signal))
      sync_t=np.zeros(len(signal))
      sync_t[index]=1
      #real_sync.append(sync_t)################################3
    #print sync_t
      for i in range(sync_count):
        sync_temp[index-i]=1
      sync_matrix.append(sync_temp)

      sync_index+=1
      sync_count=1;
    else:
      sync_count+=1
      sync_temp=np.zeros(len(signal))
      sync_matrix.append(sync_temp)
      #real_sync.append(sync_temp)##################################

    print sync_count

    energy.append(sum)
    #sample_matrix.append(temp)
    new_matrix.append(tep)
    #print now,now-sync_index*packet_len,sync_index,now-sync_index*packet_len>0,now-sync_index*packet_len<=delay
  #sync_matrix=0
  return energy,new_matrix,sync_matrix





def switch_sense(sync_start,start,packet_len,signal,magnification,granularity,delay,output_granularity):
  # new version
  # this function would sample on "Signal" with "delay". Each sample would be gotten by sensing energy and switching channels
  # The metric of granularity and delay is millisecond
  index=start
  #packet_len=0.7
  mask=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
  energy=[]
  sample_matrix=[]
  new_matrix=[]
  column=14*len(signal)*granularity/(magnification*output_granularity);
  now=0
  sampled={}
  sync_index=sync_start
  sync_matrix=[]
  sync_count=0
  real_sync=[]
  while now<(len(signal)*granularity-delay+start*delay):
    
    now=index*delay
    temp=np.zeros(column)
    tep=np.zeros(14)
    channels=mask[index%11:index%11+4]
    sum=0
    index+=1
    output_index=int((now-start*delay)/(magnification*output_granularity))
    if sampled.get(output_index)==None:
      sampled[output_index]=1
    else:
      if sampled[output_index]<11:
        sampled[output_index]+=1
      else:
        continue

    for ZigBee_channel in channels: 
      sum+=signal[int((now-start*delay)/granularity)][ZigBee_channel]
      temp[14*int((now-start*delay)/(magnification*output_granularity))+ZigBee_channel]=1
      tep[ZigBee_channel]=1
    
    # construct sync matrix 
    # if now-sync_index*packet_len>0 and now-sync_index*packet_len<delay+0.001:
    #   #sync_temp=np.zeros(len(signal))
    #   #sync_t=np.zeros(len(signal))
    #   #sync_t[index]=1
    #   #real_sync.append(sync_t)################################3
    #   #print sync_t
    #   for i in range(sync_count):
    #     sync_temp[index-i]=1
    #   sync_matrix.append(sync_temp)

    #   sync_index+=1
    #   sync_count=1;
    # else:
    #   sync_count+=1
    #   #sync_temp=np.zeros(len(signal))
    #   #sync_matrix.append(sync_temp)
    #   #real_sync.append(sync_temp)##################################

    print index

    energy.append(sum)
    #sample_matrix.append(temp)
    new_matrix.append(tep)
    #print now,now-sync_index*packet_len,sync_index,now-sync_index*packet_len>0,now-sync_index*packet_len<=delay
  sync_matrix=0
  return energy,new_matrix,sync_matrix



def refine(signal,threshold):
  result=[]
  if len(signal)>14:
    aa=np.matrix(signal)
    aa=aa.transpose()
    print aa.shape
    signal=aa.tolist()

  for ss in signal:
    temp=[]
    segment=[]
    index=0
    while index < len(ss):
      print index
      if ss[index]>0.0001:
        segment.append(1)
        index+=1

      else:
        if len(segment)==0:
          temp.append(0)
          index+=1

        else: 
          if len(segment)<=20 and len(segment)>threshold:
            for item in range(20):
              temp.append(1)
            index+=20-len(segment);
            segment=[]
            continue 
          elif len(segment)>20:
            for item in segment:
              temp.append(item)
            index+=len(segment)
            segment=[]
            continue;
          else:
            print len(segment)
            for item in segment:
              temp.append(0)
            index+=len(segment)
            segment=[]
            continue;
      
    #print tempju
    result.append(temp[:])
    #print np.matrix(result).shape
  return result

def transform(signal,length):
  data=[]
  for j in range(min(signal[0],length)):
    temp=[]
    for i in range(len(signal)):     
      temp.append(signal[i][j])
    data.append(temp)
  return data
def clear(signal,threshold,packet_len):
  result=[]
  if len(signal)>14:
    aa=np.matrix(signal)
    aa=aa.transpose()
    print aa.shape
    signal=aa.tolist()


  for ss in signal:
    temp=[]
    index=0
    s_temp=[]
    while index < len(ss):

      if index%packet_len==0 and index!=0:

        if sum(s_temp)>threshold*packet_len:
          print sum(s_temp)
          temp.append(1)

        else:
          temp.append(0)
        s_temp=[]

      s_temp.append(ss[index])
      index+=1
    result.append(temp)
  return result


def getData(Folds=[17],repeate=3,signal=[1]):
  Data={}
  temp=[]
  error=0

  # res restores all mixed signal which on data slots
  res=[]
  rr=0
  useful_energy={}
  y=[]
  ff={}
  another=[]
  count=0;
  output={}
  data_index={}
  pure_energy={}
  for item in signal:
    if item>0.01:
      y.append(1)
    else:
      y.append(0)
  for multi in Folds:
    Data[multi]={}
    data_index[multi]={}
    pure_energy[multi]={}
    output[multi]=[]
    ff[multi]=[]
    useful_energy[multi]=[]
    #start folding
    start=0
    count=0

    while start<len(y)-repeate*multi:
      #start folding:
      #print start,len(y)-repeate*multi
      folding=np.zeros(multi)
      temp=[]
      signal_index=[]
      for ii in range(repeate):
        for i in range(multi):
          #useful_energy[multi].append(0)
          signal_index.append(len(temp))
          temp.append(y[start+ii*multi+i])

        # compute folding sum
        folding+=np.array(temp)
        temp=[]
      #find out CTC data

      fold_temp=[]
      for index in range(multi):
        #ff[multi].append(folding[index])
        if folding[index]==repeate:

          #mark time slots whose sum is equal to repeate
          fold_temp.append(index)

          

      if len(fold_temp)>=2:

        # compute CTC data
        d=fold_temp[1]-fold_temp[0]


        if d<multi/2:
          if Data[multi].get(d)==None:
            Data[multi][d]=1
          else:
            Data[multi][d]+=1
        else:
          if Data[multi].get(multi-d)==None:
            Data[multi][multi-d]=1
          else:
            Data[multi][multi-d]+=1
        if d<multi/2:
          if data_index[multi].get(d)==None:
              data_index[multi][d]=[start]
          else:
            data_index[multi][d].append(start)
        else:
          if data_index[multi].get(multi-d)==None:
            data_index[multi][multi-d]=[start]
          else:
            data_index[multi][multi-d].append(start)
      else:
        error+=1
      #start+=repeate*multi
      start+=1;
      count+=1;

      # get data and raw signal
      if count%(repeate*multi)==0:
        temp_start=start;
        #print count,repeate*multi,Data[multi]
        for (dd,num) in Data[multi].items():
          data_index[multi][dd].sort()
          #print dd,num,(repeate*multi)/2,multi,dd,data_index[multi][dd]
          start=data_index[multi][dd][0]

          if num>=(repeate*multi)/2:
            #print "interval is "+str(multi)+" and data is "+str(dd)+" number of data is "+str(num)

            # dd is tha data of this a few repeat folding slots
            output[multi].append(dd)
            signal_index=[]

            folding=np.zeros(multi)
            temp=[]
            for ii in range(repeate):
              for i in range(multi):
                #useful_energy[multi].append(0)
                signal_index.append(len(temp))
                temp.append(y[start+ii*multi+i])
                useful_energy[multi].append(0)
            # get energy values on these time slots
            #print signal_index
              # compute folding sum
              folding+=np.array(temp)
              temp=[]
            #find out CTC data

            fold_temp=[]
            for index in range(multi):
              ff[multi].append(folding[index])
              if folding[index]==repeate:

                #mark time slots whose sum is equal to repeate
                fold_temp.append(index)
            print "fold_temp is "+str(fold_temp) +" and interval is"+str(multi)+" data is "+str(dd)
            # for might_index in range(len(signal_index)):
            #   if signal_index[might_index]==fold_temp[0]:
            #     #print multi,fold_temp,start,start+might_index,len(useful_energy[multi])
            #     useful_energy[multi][start+might_index]=signal[start+might_index]
            #     if pure_energy[multi].get(dd)==None:
            #       pure_energy[multi][dd]=[signal[start+might_index]]
            #     else:
            #       pure_energy[multi][dd].append(signal[start+might_index])
            #   if signal_index[might_index]==fold_temp[1]:
            #     #print might_index,fold_temp[1]
            #     useful_energy[multi][start+might_index]=signal[start+might_index]
            #     if pure_energy[multi].get(dd)==None:
            #       pure_energy[multi][dd]=[signal[start+might_index]]
            #     else:
            #       pure_energy[multi][dd].append(signal[start+might_index])
        Data[multi]={}
        data_index[multi]={}
        start=temp_start
        #print multi

# minimun energy value of consecutive  

  return Data,output,error

def eliminate(signal,threshold):
  result=[]
  if len(signal)>14:
    aa=np.matrix(signal)
    aa=aa.transpose()
    print aa.shape
    signal=aa.tolist()  
  len_list=[]
  for ss in signal:
    temp_len={}
    temp=[]
    index=0
    tt_l=0
    packet_index=0
    while index <len(ss):
      
      if ss[index]>0.001:
        temp.append(1)
        if tt_l==0:
          temp_len[index]=0;
          packet_index=index;
        tt_l+=1;

      else:
        temp.append(0)
        if tt_l>0:
          temp_len[packet_index]=tt_l
          tt_l=0
      index+=1;
    len_list.append(temp_len)
    for (ii,ll) in temp_len.items():
      if ll<threshold:
        for iii in range(ll):
          temp[ii+iii]=0

    result.append(temp)
  return result,len_list







ZigBee=generate_packets(user_num=cal(5),packet_len=cal(1),how_long=5,grind=20)
signal=ZigBee[:10000]
end=0
energy,new,sync_matrix1=switch_sense(sync_start=0,start=end,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)
energy=np.matrix(energy)
# moving packet window
energy=energy.transpose()
e1=np.ones(11)
#e1=e1.transpose()
e1=np.matrix(e1)
e1.shape
x=Variable(14,1)
diff=[]
error=[]

potential=[]

result=[]





for i in range(len(signal)-11):
  print i
  temp_new=np.matrix(new[i:i+11])
  objective=Minimize(norm1(diag(temp_new*x*e1)-energy[i:i+11]))
  constraints=[x>=0]

  prob=Problem(objective,constraints)
  differnece=prob.solve()
  diff.append(differnece)
  print i,differnece
  result.append(x.value.transpose().tolist()[0])

  #a=np.matrix.diagonal(new[i:i+20]*result*e1)
  #aa=energy[i:i+20].transpose()

# difference=(a-aa).tolist()
# error.append(difference[0])
  #plt.figure()
  #plt.imshow(x.value*e1)
  potential.append((x.value*e1).transpose().tolist())


## weave to form reconstructed signal ww. Literally, www is a list here.
ww=weave(potential,diff)
ww.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
original=np.matrix(ZigBee).transpose().tolist()
reconstruct=np.matrix(ww).transpose().tolist()

# f, axarr = plt.subplots(2, sharex=True)

# axarr[0].imshow(original)

# axarr[0].set_title("channel "+str(i)+' original signal')

# #axarr[1].imshow(rr)
# axarr[1].set_title("channel "+str(i)+" refined reconstructed signal")

# axarr[1].imshow(reconstruct)

# plt.show()
refined=refine(ww,3)
#refined=clear(ww,0.05,20)
sig_row,ss=eliminate(signal,4)
#sig_refine=clear(signal,0.05,20)
rrr,ss=eliminate(reconstruct,4)

for i in range(3):

 # plt.plot(sig_refine)
  

  plt.plot(sig_row[i],'-r')
  plt.plot(rrr[i],'-b')
  plt.title("channel "+str(i)+' reconstructed signal')
  plt.figure()
plt.show()
for i in range(3):

  #plt.plot(sig_refine)
  

  # plt.plot(sig_row[i],'-r')
  # plt.plot(reconstruct[i],'-b')
  # plt.title("channel "+str(i)+' reconstructed signal')
  # plt.figure()

  f, axarr = plt.subplots(2, sharex=True)

  axarr[0].plot(sig_row[i],'-r')
  axarr[0].plot(reconstruct[i],'-b')
  axarr[0].set_title("channel "+str(i)+' computing several sub-problems and combine the results')

  #axarr[1].imshow(rr)
  axarr[1].set_title("channel "+str(i)+' direct computing the optimization problem')

  axarr[1].plot(sig_row[i],'-r')
  axarr[1].plot(res[i],'-b')
plt.show()

for i in range(14):

  #plt.plot(sig_refine)
  f, axarr = plt.subplots(2, sharex=True)

  axarr[0].plot(sig_row[i],'-r')
  axarr[0].plot(reconstruct[i],'-b')
  axarr[0].set_title("channel "+str(i)+' reconstructed signal')

  #axarr[1].imshow(rr)
  axarr[1].set_title("channel "+str(i)+' refined reconstructed signal')

  axarr[1].plot(sig_refine[i],'-r')
  axarr[1].plot(refined[i],'-b')
plt.show()



cc=clear(signal,0.002,20)
refined=clear(ww,0.002,20)

# plt.plot(cc[0])
# plt.plot(refined[0])



# f, axarr = plt.subplots(2, sharex=True)

# axarr[0].bar(range(len(cc[3])),cc[3])

# axarr[0].set_title("original signal")

# #axarr[1].imshow(rr)
# axarr[1].set_title("refined reconstructed signal")

# axarr[1].bar(range(len(refined[3])),refined[3])

# plt.show()

for i in range(3):

  #plt.plot(sig_refine)
  f, axarr = plt.subplots(2, sharex=True)

  axarr[0].bar(range(len(cc[i][:500])),cc[i][:500])

  axarr[0].set_title("channel "+str(i)+' original signal')

  #axarr[1].imshow(rr)
  axarr[1].set_title("channel "+str(i)+" refined reconstructed signal")

  axarr[1].bar(range(len(refined[i][:500])),refined[i][:500])

plt.show()


intervals=[53,59,61,67]

data,output,error=getData(intervals,6,cc[0])


cc=clear(signal,0.003,20)
refined=clear(ww,0.003,20)

data1,output,error=getData(intervals,6,cc[0])
data2,output,error=getData(intervals,6,refined[0])

for i in range(14):
  data1,output,error=getData(intervals,6,cc[i])
  data2,output,error=getData(intervals,6,refined[i])
  print data1
  print data2












x=Variable(14,1)
objective=Minimize(norm1(diag(new*x*e1)-energy[i:i+11]))
constraints=[x>=0]

prob=Problem(objective,constraints)
differnece=prob.solve()
diff.append(differnece)



f, axarr = plt.subplots(3, sharex=True)

axarr[0].imshow(reconstruct)

axarr[0].set_title("channel "+str(i)+' original signal')

#axarr[1].imshow(rr)
axarr[1].set_title("channel "+str(i)+" refined reconstructed signal")

axarr[1].imshow(rrr)

axarr[2].set_title("channel "+str(i)+" refined reconstructed signal")

axarr[2].imshow(np.matrix(signal).transpose())
plt.show()
