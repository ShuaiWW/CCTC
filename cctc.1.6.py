import numpy as np
import matplotlib.pyplot as plt

prime=[13,17,19,23,29,31,37,41,43,47,53,59,61]


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
						for might_index in range(len(signal_index)):
							if signal_index[might_index]==fold_temp[0]:
								#print multi,fold_temp,start,start+might_index,len(useful_energy[multi])
								useful_energy[multi][start+might_index]=signal[start+might_index]
								if pure_energy[multi].get(dd)==None:
									pure_energy[multi][dd]=[signal[start+might_index]]
								else:
									pure_energy[multi][dd].append(signal[start+might_index])
							if signal_index[might_index]==fold_temp[1]:
								#print might_index,fold_temp[1]
								useful_energy[multi][start+might_index]=signal[start+might_index]
								if pure_energy[multi].get(dd)==None:
									pure_energy[multi][dd]=[signal[start+might_index]]
								else:
									pure_energy[multi][dd].append(signal[start+might_index])
				Data[multi]={}
				data_index[multi]={}
				start=temp_start
				#print multi

# minimun energy value of consecutive  

	return Data,useful_energy,error,pure_energy


def channelRebuild(signal_node,howlong):
# actually, pure_energy makes none contribution to this function

	signal=[] 
	interval=signal_node.keys()
	interval.sort()
	for inter in interval:
		signal_interval=[]
		each_energy=signal_node[inter]

		index=howlong
		while (index<=len(each_energy)):
			
			temp_energy=[]
			for slot_energy in each_energy[index-howlong:index]:
				if slot_energy>0 :
					temp_energy.append(slot_energy)
			this_energy=min(temp_energy)
			for slot_energy in each_energy[index-howlong:index]:
				if slot_energy>0 :
					signal_interval.append(this_energy)
				else:
					signal_interval.append(0)
			index+=howlong
		print index
		if len(signal)==0:
			signal=signal_interval[:]
		else:
			print len(signal),len(signal_interval),inter
			for i in range(min(len(signal),len(signal_interval))):
				if signal[i]==0 and signal_interval[i]!=0:
					signal[i]=signal_interval[i]
	#now let's combine these signals on this channel

	return signal



def getChannel(interval,signal):
	# interval is a dictionary consisting of interval set of 4 ZigBee channels
	# signal is received energy.
	# res restores energy information of each Zigbee nodes on this channel.

	channel={}
	for channel_index in interval.keys():
		data,res,error,pure_energy=getData(interval[channel_index],6,signal)
		channel[channel_index]=channelRebuild(res,100)

	return channel
		#print data


			# compute the minimun value of each 12 consecutive energy values and treate this value as the energy value during this period

def vector_distance(v1,v2):
	sum=0
	count=0
	error=[]
	available=0
	for i in range(min(len(v1),len(v2))):
		if v1[i]!=0:
			available+=1
		if v1[i]-v2[i]==0 and v1[i]!=0:
			count+=1;
		elif v1[i]-v2[i]!=0 and v1[i]!=0:
			error.append(i)

		sum+=abs(v1[i]-v2[i])

	return sum,sum/min(len(v1),len(v2)),float(count)/min(len(v1),len(v2)),count,error,float(count)/available



def vector_distance(v1,v2):
	sum=0
	count=0
	error=[]
	available=0
	for i in range(min(len(v1),len(v2))):
		if v1[i]!=0:
			available+=1
		if v1[i]-v2[i]==0 and v1[i]!=0:
			count+=1;
		elif v1[i]-v2[i]!=0 and v1[i]!=0:
			error.append(i)

		sum+=abs(v1[i]-v2[i])

	return sum,sum/min(len(v1),len(v2)),float(count)/min(len(v1),len(v2)),count,error,float(count)/available



def vector_distance(v1,v2):
	sum=0
	count=0
	error=[]
	available=0
	for i in range(min(len(v1),len(v2))):
		if (v1[i]!=0 and v2[i]!=0) :# or (v1[i]==0 and v2[i]==0):
			available+=1
			count+=1
		else:
			error.append(i)

		

	return min(len(v1),len(v2)),count

def matrix_distance(m1,m2):
	import numpy as np

	z1=np.matrix(m1)
	z1=z1.transpose()
	z1=z1.tolist()

	z2=np.matrix(m2)
	z2=z2.transpose()
	z2=z2.tolist()

	for i in range(len(z1)):
		print vector_distance(z1[i],z2[i])






def cctc(signal):
# return cross-technology communication data extracted from energy information received by Wi-Fi device

# 1st phase, solve equations and get energy information in each ZigBee chanel

# 2nd phase, Demodulation on each Zigbee channels and get CTC data

	return 0;


def generate_multiplex(intervals,howlong=5,num=1,data=3):
	import numpy as np
	RSSI_channel=[]
	
	temp=np.zeros(1000*howlong)
	for i in range(num):
		start=np.random.randint(11);
		signal_energy=np.random.random_sample()*0.01
		for j in range(1000*howlong):
			if j>=start:
				if (j-start)%intervals[i]==0:
					if temp[j]==0:
						temp[j]=signal_energy+0.01;
				if (j-start-data)%intervals[i]==0:
					if temp[j]==0:
						temp[j]=signal_energy+0.01;
	return list(temp)


def generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5):
	# user_num is number of users on each ZigBee channel
	# packet_len is the length of each Zigbee channel
	# how_long is how long time this simulation runs
	RSSI=[]
	intervals=[11,13,61,67,71,73,79,83];
	interval1=[17,31,47,89,109,101,103,107]
	interval2=[19,37,53,97,139,127,131,137]
	interval3=[23,43,59,113,149,151,157,163]
	#starts=np.random.randint(interval[0],size=14)
	#start=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	data=[3,3,3,3,3,3,3,3,3,3,3,4,5,6]
	#data=np.random.randint(5,size=16)
	RSSI_channel=[]
	for channel in range(len(user_num)-3):
		num=user_num[channel]
		temp=np.zeros(1000*how_long)
		for i in range(num):
			signal_energy=np.random.random_sample()*0.01
			start=np.random.randint(11);
			for j in range(1000*how_long):
				if j>=start:
					if (j-start)%intervals[i]==0:
						if temp[j]==0:
							temp[j]=signal_energy+0.01;
					if (j-start-data[i])%intervals[i]==0:
						if temp[j]==0:
							temp[j]=signal_energy+0.01;
		RSSI_channel.append(list(temp))

	RSSI_channel.append(generate_multiplex(intervals=interval1,num=user_num[11],howlong=how_long,data=data[11]))
	RSSI_channel.append(generate_multiplex(intervals=interval2,num=user_num[12],howlong=how_long,data=data[12]))
	RSSI_channel.append(generate_multiplex(intervals=interval3,num=user_num[13],howlong=how_long,data=data[13]))
	#return RSSI_channel
	for time in range(1000*how_long):
		temp=[];
		for i in range(len(user_num)):
			temp.append(RSSI_channel[i][time])
		RSSI.append(temp)
	return RSSI

# interval1=[17,31,47,89,109,101,103,107]
# data=[3,3,3,3,3,3,3,3,3,3,3,3,3,3]
# user_num=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,];
# packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,]
# how_long=5
# a=generate_multiplex(intervals=interval1,num=user_num[11],howlong=how_long,data=data[11])

def wifi(ZigBee):
	length=len(ZigBee[0])
	Wi=[]
	for time in range(len(ZigBee)):
		temp=[]
		for i in range(length-3):
			temp.append(ZigBee[time][i]+ZigBee[time][i+1]+ZigBee[time][i+2]+ZigBee[time][i+3])
		Wi.append(temp)
	return Wi


ZigBee=generate_packets(user_num=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5)


ZigBee=generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5)


ZigBee=generate_packets(user_num=[2,2,2,2,2,2,2,2,2,2,2,2,2,2],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5)
ZigBee=generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,2,2,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5)
wi=wifi(ZigBee)
multiplex=np.matrix(wi)
temp=multiplex.transpose()
multiplex=temp.tolist()[10]

intervals=[11,13,61,67,71,73,79,83];
interval1=[17,31,47,89,109,101,103,107]
interval2=[19,37,53,97,139,127,131,137]
interval3=[23,43,59,113,149,151,157,163]
interval={1:intervals,2:interval1,3:interval2,4:interval3}


########################################################################################
ZigBee=generate_packets(user_num=[1,1,1,1,1,1,1,1,1,1,1,2,2,1],packet_len=[1,1,1,1,1,1,1,1,1,1,1,1,1,1],how_long=5)
wi=wifi(ZigBee)
multiplex=np.matrix(wi)
temp=multiplex.transpose()
multiplex=temp.tolist()[10]
Data,res,error,ff=getData(interval3,6,multiplex)

si=channelRebuild(res,100)
temp=np.matrix(ZigBee)
ZZ=temp.transpose()
ZZ=ZZ.tolist()[13]
print distance(si,ZZ)
getChannel(interval,multiplex)



a=generate_multiplex(intervals,howlong=5,num=3,data=3)
b=generate_multiplex(interval2,howlong=5,num=3,data=3)
c=np.array(a)+np.array(b)
multiplex=list(c)
getChannel(interval,multiplex)




intervals=[11,13,61,67,71,73,79,83];
interval1=[23,43,59,113,149,151,157,163,11,13,61,67,71,73,79,83,17,31,47,89,109,101,103,107,19,37,53,97,139,127,131,137]
interval2=[19,37,53,97,139,127,131,137]
interval3=[23,43,59,113,149,151,157,163]

interval1=[23,43,59,113,149,151,157,163,11,13,61,67,71,73,79,83,17,31,47,89,109,101,103,107,19,37,53,97,139,127,131,137]
interval1.sort()
multiplex=generate_multiplex(intervals=interval1,num=8,howlong=5,data=3)
Data,res,error,ff=getData(interval1,5,multiplex)


raw=np.matrix(ZigBee)
temp=raw.transpose()
raw=temp.tolist()[12]




new=[]
signal=wi


signal_len=min(len(channel_11_14[1]),len(channel_11_14[2]),len(channel_11_14[3]),len(channel_11_14[4]))

for i in range(signal_len):
	wifi=signal[i]
	temp=[0,0,0,0,0,0,0,0,0,0,channel_11_14[1][i],channel_11_14[2][i],channel_11_14[3][i],channel_11_14[4][i]]
	print len(wifi),len(temp)

	for i in range(len(wifi)-1):
		channel_num=len(wifi)-1
		temp[channel_num-i]=wifi[channel_num-i]-temp[channel_num-i+1]-temp[channel_num-i+2]-temp[channel_num-i+3]-temp[channel_num-i]
	new.append(temp)
return ZigBee



m1=ZigBee
m2=new_zigbee
z1=np.matrix(m1)
z1=z1.transpose()
z1=z1.tolist()

z2=np.matrix(m2)
z2=z2.transpose()
z2=z2.tolist()

for i in range(len(z1)):
	print vector_distance(z1[i],z2[i])
