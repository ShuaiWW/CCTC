import numpy as np
import matplotlib.pyplot as plt
dd=[]

for line in open("WiFisoft.txt"):
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
