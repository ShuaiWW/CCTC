import numpy as np

import matplotlib.pyplot as plt

from cvxpy import *


def equal(v1,v2):
	if (v1-v2)<0.02 and (v1-v2)>-0.02:
		return True;
	else:
		return False
def energy(signal):
	ll={}
	t_length=0
	t_signal=[signal[0]];
	for index in range(1,len(signal)):


		if equal(signal[index-1],signal[index])==True:
			#print signal[index-1],signal[index],t_length,t_signal
			t_length+=1;
			t_signal.append(signal[index])
		else:
			#print t_length,t_signal
			ll[t_length]=t_signal
			t_length=0
			t_signal=[]
	ll[t_length]=t_signal
	#print ll
	if ll.items()==[]:
		return np.mean(signal)
	else:
		aa=sorted(ll.keys(),reverse=True)
		#if equal(np.mean(ll[aa[0]]),0)==True:
		return np.mean(ll[aa[0]])

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

def sense_matrix(channels):
	sequence=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
	mask=[0.5,1,0.95,0.5]
	channel_map={2412000000:0,2417000000:1,2422000000:2,2427000000:3,2432000000:4,2437000000:5,2442000000:6,2447000000:7,2452000000:8,2457000000:9,2462000000:10}
	result=[]
	for cc in channels:
		current=channel_map[cc]
		temp=np.zeros(14)
		zigbee_channel=sequence[current:current+4]
		for index in range(4):
			temp[zigbee_channel[index]]=mask[index]
		result.append(temp)
	return result


def weave(potential,error,threshold):
  result=[]
  result=potential[0][:]
  temp_plus=0
  for i in range(1,len(error)):
    print i 
    if error[i]<=threshold:
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



def ext(delay,packet_len,signal_len):
  row_num=int(signal_len/(packet_len/delay))
  result=[]
  for i in range(row_num):
    temp=np.zeros(signal_len)
    for j in range(int(packet_len/delay)):
      temp[i*int(packet_len/delay)+j]=1
    result.append(temp)
  return np.matrix(result)


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
        if folding[index]>=(repeate/2):

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




filename="final.txt"
x=[]
y=[]
tt=[]
con=[]
lll=open(filename)
content=lll.readline().split(",")
channel=[]
sec=float(content[0])
mic=float(content[1])
pre=sec+mic
sample=[]
for line in open(filename):
	content=line.split(",")
	sec=float(content[0])
	mic=float(content[1])
	temp_channel=int(float(content[2]))
	now=sec+mic
	if now==pre:
		continue;
	left=[]
	length=len(content)-3
	for i in range(3,len(content)):
		left.append(float(content[i]))
		sample.append(float(content[i]))
		tt.append(i*((now-pre)/length)+pre)

	con.append([now,left])
	pre=now

	x.append(now)
	channel.append(temp_channel)

# last=x[0]
# diff=[]
# for index in range(1,len(x)):
# 	current=x[index]
# 	diff.append(current-last)
# 	last=current;

# sum1=[]
# last=diff[0]
# for index in range(1,len(diff)):
# 	current=diff[index]
# 	sum1.append(current+last)
# 	last=current;
# plt.plot(sum1,'-b*')
# plt.show()

# plt.hist(sum1,bins='auto',normed=True)
# plt.title("histogram of Delay")
# plt.show()

sensed_energy=[]
x_sense=[]
for ii in con:
	t_energy=energy(ii[1])
	if equal(t_energy,0)==True:
		sensed_energy.append(0)
	else:
		sensed_energy.append(t_energy)
	x_sense.append(ii[0])




#################  Asynchronous version of original signal:
##################################################################
new=sense_matrix(channel)





ee=ext(0.35,4,1000)
x=Variable(14,87)
temp_new=np.matrix(new[6000:7000])
objective=Minimize(norm1(diag(temp_new*x*ee)-sensed_energy[6000:7000]))
constraints=[x>=0]

prob=Problem(objective,constraints)
differnece=prob.solve()
#plt.imshow(x.value*ee)
#plt.title("all ")
#plt.show()

signal=x.value.tolist()




##################################################################3



e1=np.ones(11)
#e1=e1.transpose()
e1=np.matrix(e1)
e1.shape
x=Variable(14,1)
diff=[]
error=[]

potential=[]

result=[]
for i in range(6000,7000-11):
  #print i
  if i%100==0:
  	print i
  temp_new=np.matrix(new[i:i+11])
  objective=Minimize(norm1(diag(temp_new*x*e1)-sensed_energy[i:i+11]))
  constraints=[x>=0]

  prob=Problem(objective,constraints)
  differnece=prob.solve()
  diff.append(differnece)
  #print i,differnece
  result.append(x.value.transpose().tolist()[0])

  #a=np.matrix.diagonal(new[i:i+20]*result*e1)
  #aa=energy[i:i+20].transpose()

# difference=(a-aa).tolist()
# error.append(difference[0])
  #plt.figure()
  #plt.imshow(x.value*e1)
  potential.append((x.value*e1).transpose().tolist())


## weave to form reconstructed signal ww. Literally, www is a list here.
ww=weave(potential,diff,1)
ww.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#original=np.matrix(ZigBee).transpose().tolist()
reconstruct=np.matrix(ww).transpose().tolist()
plt.figure()
plt.imshow(reconstruct)
plt.title("threshold=1")

ww=weave(potential,diff,0.1)
ww.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#original=np.matrix(ZigBee).transpose().tolist()
reconstruct=np.matrix(ww).transpose().tolist()
plt.figure()
plt.imshow(reconstruct)
plt.title("threshold=0.1")
plt.show()


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
plt.imshow(rrr)
plt.title("windows")
plt.show()



import socket
import time
s=socket.socket()
dest = ('<broadcast>', 51423)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
while True:

	s.sendto("00000000000000000000000000000000000000000000000000000000000000000000000000000000000", dest)
	time.sleep(0.005)







import numpy as np
import matplotlib.pyplot as plt
dd=[]

for line in open("WiFisoft_80211b.txt"):
  data=line.split(',')
  for i in range(len(data)):
    if data[i]!=' ':
      dd.append(float(data[i]))

soft1=dd[82291:90549]
soft2=dd[41452:49710]
plt.plot(soft1,'-bs')
plt.figure()
plt.plot(soft2,'-r*')
plt.figure()
plt.plot(dd)
plt.show()
