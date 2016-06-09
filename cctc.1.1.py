import numpy as np
import matplotlib.pyplot as plt

prime=[13,17,19,23,29,31,37,41,43,47,53,59,61]



def getData(Folds=[17],repeate=3,signal=[1]):
	Data={}
	temp=[]
	error=0
	res=[]
	rr=0
	useful_energy={}
	y=[]

	for item in signal:
		if item>0.01:
			y.append(1)
		else:
			y.append(0)
	for multi in Folds:
		Data[multi]={}
		useful_energy[multi]=[]
		#start folding
		start=0

		while start<len(y)-repeate*multi:
			#start folding:
			
			folding=np.zeros(multi)
			temp=[]
			signal_index=[]
			for ii in range(repeate):
				for i in range(multi):
					useful_energy[multi].append(0)
					signal_index.append(len(temp))
					temp.append(y[start+ii*multi+i])


				# compute folding sum
				folding+=np.array(temp)
				temp=[]
			#find out CTC data

			fold_temp=[]
			for index in range(multi):
				ff.append(folding[index])
				if folding[index]==repeate:

					#mark time slots whose sum is equal to repeate
					fold_temp.append(index)

					# get energy values on these time slots
					for might_index in range(len(signal_index)):
						if signal_index[might_index]==index:
							useful_energy[multi][start+might_index]=signal[start+might_index]
					

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
			else:
				error+=1
			start+=repeate*multi

	return Data,useful_energy,error

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
					temp[j]=signal_energy+0.01;
				if (j-start-data)%intervals[i]==0:
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
	data=[3,3,3,3,3,3,3,3,3,3,3,3,3,3]
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
						temp[j]=signal_energy+0.01;
					if (j-start-data[i])%intervals[i]==0:
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
wi=wifi(ZigBee)
multiplex=np.matrix(wi)
temp=multiplex.transpose()
multiplex=temp.tolist()[10]


Data,res,error=getData([17,19,23],3,multiplex)


