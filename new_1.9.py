

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
	


ZigBee=generate_packets(user_num=cal(10),packet_len=cal(1),how_long=5,grind=20)

large=shrink(signal=ZigBee[:1200],granularity=0.25,packet_len=3,output_granularity=1)

ss=shrink(signal=ZigBee[:300],granularity=0.05,packet_len=1,output_granularity=1)

delay=0.25
result=[]
ll=0

ZigBee=generate_packets(user_num=cal(5),packet_len=cal(1),how_long=5,grind=20)

signal=ZigBee[900:1000]

mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

zig_sync=np.matrix(signal)
zig_sync=zig_sync.transpose()
energy=energy.transpose()

z=[]
sample_num=20
for i in range(sample_num):
	z.append(Variable(14,100))
x=Variable(14,100)

right_matrix=[]
objective=Minimize(norm1(x-z[1])+norm1(x-z[2])+norm1(x-z[3])+norm1(x-z[4])+norm1(x-z[5])+norm1(x-z[6])+norm1(x-z[7])+norm1(x-z[8])+norm1(x-z[9])+norm1(x-z[10])+norm1(x-z[11])+norm1(x-z[12])+norm1(x-z[13])+norm1(x-z[14])+norm1(x-z[15])+norm1(x-z[16])+norm1(x-z[17])+norm1(x-z[18])+norm1(x-z[19])+norm1(x-z[0]))
#constraints=[diag(new*x)==energy,0<=x,norm1(x-aa)<=0.5]
constraints=[diag(new*x)==energy,0<=x]
for i in range(sample_num):
	sync_matrix,real_sync=generate_sync(start=i,sample_num=20,length=100)
	right_matrix.append(real_sync)
	constraints.append(z[i]>=0)
	constraints.append(z[i]==x*sync_matrix)
	constraints.append(sum_entries(x)==sum_entries(z[i]))
#	for j in range(14):
#		rr=np.zeros((14,14))
#		rr[j,j]=1;		
#		constraints.append(sum_entries(rr*x)==sum_entries(rr*z[i]))


constraints.append(x==z[19]*right_matrix[19]+z[18]*right_matrix[18]+z[17]*right_matrix[17]+z[16]*right_matrix[16]+z[15]*right_matrix[15]+z[14]*right_matrix[14]+z[13]*right_matrix[13]+z[12]*right_matrix[12]+z[11]*right_matrix[11]+z[10]*right_matrix[10]+z[9]*right_matrix[9]+z[8]*right_matrix[8]+z[7]*right_matrix[7]+z[6]*right_matrix[6]+z[5]*right_matrix[5]+z[4]*right_matrix[4]+z[3]*right_matrix[3]+z[2]*right_matrix[2]+z[1]*right_matrix[1]+z[0]*right_matrix[0])


#constraints=[diag(new*x)==energy,0<=x,z>=0,z==x*sync_matrix]
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

aa=x.value.tolist()
bb=z.value.tolist()
s1=0
for i in range(14):
	s1+=sum(aa[i])
s2=0
for i in range(14):
	s2+=sum(bb[i])

print s1,s2

s_original=0

for i in range(14):
	s_original+=sum(aa[i])
s2=0
for i in range(14):
	s2+=sum(bb[i])

plt.show()
print diff(ss,vv)

mat,energy,new,sync_matrix=switch_sense(sync_start=0,start=0,signal=ZigBee[:200],magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

zig_sync=np.matrix(ZigBee[:200])
zig_sync=zig_sync.transpose()
energy=energy.transpose()

signal=ZigBee[900:1100]
recon=np.zeros((14,200))
for i in range(20):

	mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

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






















z=[]
right_matrix=[]
sample_num=20

x=np.matrix(signal).transpose()


for i in range(sample_num):
	sync_matrix,real_sync=generate_sync(start=i,sample_num=20,length=100)
	right_matrix.append(real_sync)
	
	z.append(x*sync_matrix)
	
#	for j in range(14):
#		rr=np.zeros((14,14))
#		rr[j,j]=1;		
#		constraints.append(sum_entries(rr*x)==sum_entries(rr*z[i]))


re=z[19]*right_matrix[19]+z[18]*right_matrix[18]+z[17]*right_matrix[17]+z[16]*right_matrix[16]+z[15]*right_matrix[15]+z[14]*right_matrix[14]+z[13]*right_matrix[13]+z[12]*right_matrix[12]+z[11]*right_matrix[11]+z[10]*right_matrix[10]+z[9]*right_matrix[9]+z[8]*right_matrix[8]+z[7]*right_matrix[7]+z[6]*right_matrix[6]+z[5]*right_matrix[5]+z[4]*right_matrix[4]+z[3]*right_matrix[3]+z[2]*right_matrix[2]+z[1]*right_matrix[1]+z[0]*right_matrix[0]



f, axarr = plt.subplots(2, sharex=True)
axarr[0].imshow(x)
axarr[0].set_title('original signal')
axarr[1].imshow(re)
axarr[1].set_title('synchronous signal')


plt.show()




###############  new idea:

def ext(delay,packet_len,signal_len):
	row_num=int(signal_len/(packet_len/delay))
	result=[]
	for i in range(row_num):
		temp=np.zeros(signal_len)
		for j in range(int(packet_len/delay)):
			temp[i*int(packet_len/delay)+j]=1
		result.append(temp)
	return np.matrix(result)



signal=ZigBee[900:1100]

extend=ext(delay=0.05,packet_len=1,signal_len=len(signal))



mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)

zig_sync=np.matrix(signal)
zig_sync=zig_sync.transpose()
energy=energy.transpose()

z=[]
sample_num=20
for i in range(sample_num):
	z.append(Variable(14,int(len(signal)/sample_num)))
x=Variable(14,len(signal))


#objective=Minimize(norm1(x-z[1])+norm1(x-z[2])+norm1(x-z[3])+norm1(x-z[4])+norm1(x-z[5])+norm1(x-z[6])+norm1(x-z[7])+norm1(x-z[8])+norm1(x-z[9])+norm1(x-z[10])+norm1(x-z[11])+norm1(x-z[12])+norm1(x-z[13])+norm1(x-z[14])+norm1(x-z[15])+norm1(x-z[16])+norm1(x-z[17])+norm1(x-z[18])+norm1(x-z[19])+norm1(x-z[0]))
objective=Minimize(norm1(z[0]))
constraints=[diag(new*x)==energy,0<=x]

right_matrix=[]

for i in range(sample_num):
	sync_matrix,real_sync=generate_sync(start=i,sample_num=20,length=len(signal))
	right_matrix.append(real_sync)
	constraints.append(z[i]>=0)
	#constraints.append(z[i]==x*sync_matrix)
	constraints.append(sum_entries(x)==sum_entries(z[i]*extend))
	if i+1<sample_num:
		constraints.append(sum_entries(z[i])==sum_entries(z[i+1]))
#	for j in range(14):
#		rr=np.zeros((14,14))
#		rr[j,j]=1;		
#		constraints.append(sum_entries(rr*x)==sum_entries(rr*z[i]))


constraints.append(x==z[19]*extend*right_matrix[19]+z[18]*extend*right_matrix[18]+z[17]*extend*right_matrix[17]+z[16]*extend*right_matrix[16]+z[15]*extend*right_matrix[15]+z[14]*extend*right_matrix[14]+z[13]*extend*right_matrix[13]+z[12]*extend*right_matrix[12]+z[11]*extend*right_matrix[11]+z[10]*extend*right_matrix[10]+z[9]*extend*right_matrix[9]+z[8]*extend*right_matrix[8]+z[7]*extend*right_matrix[7]+z[6]*extend*right_matrix[6]+z[5]*extend*right_matrix[5]+z[4]*extend*right_matrix[4]+z[3]*extend*right_matrix[3]+z[2]*extend*right_matrix[2]+z[1]*extend*right_matrix[1]+z[0]*extend*right_matrix[0])


#constraints=[diag(new*x)==energy,0<=x,z>=0,z==x*sync_matrix]
#constraints=[norm1(z)==norm1(x),z>=0]
#constraints=[mat*x==energy]
prob=Problem(objective,constraints)
prob.solve()

f, axarr = plt.subplots(3, sharex=True)

a=np.matrix(signal)

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(zig_sync*sync_matrix)
axarr[1].set_title('synchronous signal')
axarr[2].imshow(x.value*extend)
axarr[2].set_title('reconstructed signal')


plt.show()


for i in range(20):
	plt.figure()
	plt.imshow(z[i].value)
plt.show()


axarr[3].imshow(z[0].value)
axarr[3].set_title('reconstructed synchronous signal')


x=Variable(14,20)
objective=Minimize(norm1(diag(new[:20]*x)-energy[:20]))
constraints=[0<=x]
prob=Problem(objective,constraints)
prob.solve()

f, axarr = plt.subplots(3, sharex=True)

a=np.matrix(signal)

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(zig_sync*sync_matrix)
axarr[1].set_title('synchronous signal')
axarr[2].imshow(x.value*extend)
axarr[2].set_title('reconstructed signal')

plt.show()


plt.imshow(x.value*extend)
plt.show()




signal=ZigBee[899:1099]

extend=ext(delay=0.05,packet_len=1,signal_len=len(signal))



mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)


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

for i in range(289):
	objective=Minimize(norm1(diag(new[i:i+11]*x*e1)-energy[i:i+11]))
	constraints=[x>=0]

	prob=Problem(objective,constraints)
	differnece=prob.solve()
	diff.append(differnece)
	print i,differnece
	result.append(x.value.transpose().tolist()[0])

	#a=np.matrix.diagonal(new[i:i+20]*result*e1)
	#aa=energy[i:i+20].transpose()

#	difference=(a-aa).tolist()
#	error.append(difference[0])
	#plt.figure()
	#plt.imshow(x.value*e1)
#	potential.append((x.value*e1).transpose().tolist())


rr=np.matrix(result)

rr=rr.transpose()

f, axarr = plt.subplots(2, sharex=True)

a=np.matrix(signal)

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(rr)
axarr[1].set_title('average')


plt.show()

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
	return result

rr=weave(potential,diff)

rr=np.matrix(result)
rr=rr.transpose()
f, axarr = plt.subplots(2, sharex=True)

a=np.matrix(signal) 

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')
axarr[1].imshow(rr)
axarr[1].set_title('difference')


plt.show()



############################################### 8.1 
# even though we could not solve exact value in original matrix correctly,
# maybe we might get average value of several consecutive samples.

original=[[0,0,0,0,1,1,1,1,0,0,0,0],[0,0,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,1,0]]
original=np.matrix(original)
s=[[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1]]
s=np.matrix(s)

w=np.diag(s*original)

c=np.matrix([1,1,1,1])

last=[]

x=Variable(3,1)
for i in range(8):
	objective=Minimize(norm1(diag(s[i:i+4]*x*c)-w[i:i+4]))
	constraints=[0<=x]
	prob=Problem(objective,constraints)
	prob.solve()
	last.append(x.value.transpose().tolist()[0])

tt=np.matrix(last)
plt.imshow(tt.transpose())
plt.show()

a=np.matrix(signal)

a=a.transpose().tolist()
plt.plot(aa[3],'*-')
plt.figure()
plt.imshow(a)
plt.show()








original=[[1,1,1,1,1,1,1,1],[1,1,0,0,0,0,0,0],[0,0,0,0,1,1,1,1]]
original=np.matrix(original)
s=[[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1],[1,1,0],[0,1,1]]
s=np.matrix(s)

w=np.diag(s*original)


c=np.matrix([1,1,1,1,1,1,1,1])
Z=Variable(3,8)
A=Variable(3,1)

#+10*
objective=Minimize(norm2(diag(s*A*c)-w))
constraints=[0<=Z,A>=0,diag(s*Z)==w,Z*c.transpose()==8*A]
prob=Problem(objective,constraints)
prob.solve()

print Z.value
print 4*A.value





mat,energy,new,sync_matrix,real_sync=switch_sense(sync_start=0,start=0,packet_len=1,signal=signal,magnification=1,granularity=0.05,delay=0.05,output_granularity=0.05)


# moving packet window
energy=energy.transpose()
e1=np.ones(11)
#e1=e1.transpose()
e1=np.matrix(e1)
e1.shape
z=Variable(14,len(signal))
diff=[]
error=[]

potential=[]

result=[]


objective=Minimize(norm1(z))
constraints=[z>=0,diag(new*z)==energy]

prob=Problem(objective,constraints)
differnece=prob.solve()

	#aa=energy[i:i+20].transpose()

#	difference=(a-aa).tolist()
#	error.append(difference[0])
	#plt.figure()
	#plt.imshow(x.value*e1)
#	potential.append((x.value*e1).transpose().tolist())



rr=np.matrix(result)
rr=rr.transpose()
f, axarr = plt.subplots(3, sharex=True)

a=np.matrix(signal) 

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')

axarr[1].imshow(rr)
axarr[1].set_title('approximate ')


axarr[2].imshow(z.value)
axarr[2].set_title('directly ')

plt.show()


f, axarr = plt.subplots(2, sharex=True)

a=np.matrix(signal) 

a=a.transpose().tolist()
axarr[0].imshow(a)
axarr[0].set_title('original signal')

axarr[1].imshow(x.value)
axarr[1].set_title('directly ')
plt.show()






objective=Minimize(norm1(diag(s[1:1+4]*x*c)-w[1:1+4]))
constraints=[0<=x]
prob=Problem(objective,constraints)
prob.solve()
	

tt=np.matrix(last)
plt.imshow(tt.transpose())
plt.show()
