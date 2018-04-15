
#统计正确率

Precisions = []
Recalls = []

for i in range(5):

	TP = 0 
	FP = 0
	FN = 0
	TN = 0

	with open(str(i)+'_result', 'r') as f:
		lines = f.readlines()
		for line in lines:
		
			if len(line) <= 0:
				continue
			
			units = line.strip().split('\t')[-2:]
			if len(units) < 2:
				# print(line)
				# print(units)
				continue
				
			if units[0] == units[1] :
				if units[0] == "S":
					TN += 1
				else:
					TP += 1
			else:
				if units[0] == "S":
					FP += 1
				else:
					FN += 1
	print(TP, FP, FN, TN)
	print("Precision:" , TP / (TP+FP))
	print("Recall:", TP/(TP+FN))
	Precisions.append(TP / (TP+FP))
	Recalls.append(TP/(TP+FN))

print('all Precision:', Precisions)
print('average Precision:',sum(Precisions)/len(Precisions))
print('all Recalls', Recalls)
print('average Recall:', sum(Recalls)/len(Recalls))
F1 = [2*Precisions[i]*Recalls[i]/(Precisions[i]+Recalls[i]) for i in range(len(Precisions))]
print ('all F1 score' , F1)
print('average F1 score:', sum(F1)/len(F1))
