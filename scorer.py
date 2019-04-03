
#key=open(r"C:\Users\alaga\Desktop\sem 2\AIT690\WSD1\line-answers.txt")


import numpy as np
import pandas as pd
import sys

my_list=sys.argv[1]
key=sys.argv[2]

my_list=open(my_list)
my_list=my_list.read()

key=open(key)
key=key.read()

key=key.split()
my_list=my_list.split()

ans_key=[]
my_ans=[]
for i in range(0,len(key)):
    if(key[i].find("senseid")==False):
        ans_key.append(key[i])
        my_ans.append(my_list[i])



initialList=np.zeros((2,2))
confusionMatrix=pd.DataFrame(initialList, index=["phone","product"], columns=["phone","product"])
acc=0
for i in range(0,len(ans_key)):
    if(ans_key[i]==my_ans[i]):
        acc+=1
        actual=(ans_key[i].split("="))[1].replace('"','')[:-2]
        predicted=(my_ans[i].split("="))[1].replace('"','')[:-2]
        count=confusionMatrix.at[actual,predicted]
        count+=1
        confusionMatrix.at[actual,predicted]=count
    else:
        actual=(ans_key[i].split("="))[1].replace('"','')[:-2]
        predicted=(my_ans[i].split("="))[1].replace('"','')[:-2]
        count=confusionMatrix.at[actual,predicted]
        count+=1
        confusionMatrix.at[actual,predicted]=count

print()
print('Confusion Matrix: ')
print(confusionMatrix)
print()
print('Accuracy: ',acc/len(ans_key))
