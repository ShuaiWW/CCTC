

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



def switch_sense(sync_start,start,signal,magnification,granularity,delay,output_granularity):
	# new version
	# this function would sample on "Signal" with "delay". Each sample would be gotten by sensing energy and switching channels
	# The metric of granularity and delay is millisecond
	index=start
	packet_len=0.7
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
	while now<len(signal)*granularity-delay:
		
		now=index*delay
		temp=np.zeros(column)
		tep=np.zeros(14)
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
			tep[ZigBee_channel]=1
		
		# construct sync matrix	
		if now-sync_index*packet_len>0 and now-sync_index*packet_len<delay+0.001:
			sync_temp=np.zeros(len(signal))
			sync_t=np.zeros(len(signal))
			sync_t[index]=1
			real_sync.append(sync_t)
			print sync_t
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
		
	return np.matrix(sample_matrix),np.matrix(energy),np.matrix(new_matrix),np.matrix(sync_matrix),np.matrix(real_sync)


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
	


ZigBee=generate_packets(user_num=cal(10),packet_len=cal(1),how_long=5,grind=20)

large=shrink(signal=ZigBee[:1200],granularity=0.25,packet_len=3,output_granularity=1)

ss=shrink(signal=ZigBee[:300],granularity=0.05,packet_len=1,output_granularity=1)

delay=0.25
result=[]
ll=0

ZigBee=generate_packets(user_num=cal(5),packet_len=cal(1),how_long=5,grind=20)

signal=ZigBee[900:1100]

mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

zig_sync=np.matrix(signal)
zig_sync=zig_sync.transpose()
energy=energy.transpose()

z=Variable(14,200)
x=Variable(14,200)


objective=Minimize(norm1(z-x))
#constraints=[diag(new*x)==energy,0<=x,norm1(x-aa)<=0.5]
constraints=[diag(new*x)==energy,0<=x,z==x*sync_matrix]

#constraints=[norm1(z)==norm1(x),z>=0]
#constraints=[mat*x==energy]
prob=Problem(objective,constraints)
prob.solve()

f, axarr = plt.subplots(4, sharex=True)

a=np.matrix(signal)

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(zig_sync*sync_matrix)
axarr[1].set_title('synchronous signal')
axarr[2].imshow(x.value)
axarr[2].set_title('reconstructed signal')
axarr[3].imshow(z.value)
axarr[3].set_title('reconstructed synchronous signal')

plt.show()
print diff(ss,vv)

mat,energy,new,sync_matrix=switch_sense(sync_start=0,start=0,signal=ZigBee[:200],magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

zig_sync=np.matrix(ZigBee[:200])
zig_sync=zig_sync.transpose()
energy=energy.transpose()


recon=np.zeros((14,200))
for i in range(17):

	mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=i*0.05,start=0,signal=ZigBee[:200],magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

	sync_matrix,real_sync=generate_sync(start=i,sample_num=20,length=200)
	energy=energy.transpose()
	zig_sync=np.matrix(ZigBee[:200])
	zig_sync=zig_sync.transpose()

	x=Variable(14,200)
	z=Variable(14,200)
	objective=Minimize(norm1(z-x))
	#constraints=[diag(new*x)==energy,0<=x]
	constraints=[diag(new*x)==energy,0<=x,z==x*sync_matrix]

	#constraints=[norm1(z)==norm1(x),z>=0]
	#constraints=[mat*x==energy]
	prob=Problem(objective,constraints)
	prob.solve()

	f, axarr = plt.subplots(4, sharex=True)

	a=np.matrix(ZigBee[:200])

	a=a.transpose().tolist()
	axarr[0].imshow(a)
	axarr[0].set_title(str(i)+' original signal')
	axarr[1].imshow(zig_sync*sync_matrix)
	axarr[1].set_title(str(i)+' synchronous signal')
	axarr[2].imshow(x.value)
	axarr[2].set_title(str(i)+' reconstructed signal')
	axarr[3].imshow(z.value)
	axarr[3].set_title(str(i)+' reconstructed synchronous signal')
	recon+=x.value*real_sync
plt.figure()
plt.imshow(recon)
plt.show()






	energy=energy.transpose()
	zig_sync=np.matrix(ZigBee[:200])
	zig_sync=zig_sync.transpose()

	x=Variable(14,200)
	z=Variable(14,200)
	objective=Minimize(norm1(z-x))
	#constraints=[diag(new*x)==energy,0<=x]
	constraints=[diag(new*x)==energy,0<=x,z==x*sync_matrix]

	#constraints=[norm1(z)==norm1(x),z>=0]
	#constraints=[mat*x==energy]
	prob=Problem(objective,constraints)
	prob.solve()

	f, axarr = plt.subplots(4, sharex=True)

	a=np.matrix(ZigBee[:200])

	a=a.transpose().tolist()
	axarr[0].imshow(a)
	axarr[0].set_title(str(i)+' original signal')
	axarr[1].imshow(zig_sync*sync_matrix)
	axarr[1].set_title(str(i)+' synchronous signal')
	axarr[2].imshow(x.value)
	axarr[2].set_title(str(i)+' reconstructed signal')
	axarr[3].imshow(z.value)
	axarr[3].set_title(str(i)+' reconstructed synchronous signal')









f, axarr = plt.subplots(2, sharex=True)

a=np.matrix(ZigBee[:200])

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')

axarr[1].imshow(x.value)
axarr[1].set_title('reconstructed signal')

plt.show()










vv=reform(14,x.value)



a=np.matrix(ZigBee[:300])

a=a.transpose().tolist()
b=np.matrix(vv)
b=b.transpose().tolist()


zig_sync=np.matrix(ZigBee[:300])
zig_sync=zig_sync.transpose()
#plt.imshow(zig_sync*sync_matrix)
#plt.show()


a=np.matrix(ZigBee[:300])
b=np.matrix(vv)
a=a.transpose().tolist()
b=b.transpose().tolist()
f, axarr = plt.subplots(2, sharex=True)
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(zig_sync*sync_matrix)
axarr[1].set_title('synchronous signal')

plt.show()
ZigBee=generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5,grind=4):
