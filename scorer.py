
#key=open(r"C:\Users\alaga\Desktop\sem 2\AIT690\WSD1\line-answers.txt")

my_list=sys.argv[1]
key=sys.argv[2]

my_list=my_list.read()
key=key.read()

key=key.split()
my_list=my_list.split()

ans_key=[]
my_ans=[]
for i in range(0,len(key)):
    if(key[i].find("senseid")==False):
        ans_key.append(key[i])
        my_ans.append(my_list[i])

acc=0
for i in range(0,len(ans_key)):
    if(ans_key[i]==my_ans[i]):
        acc+=1


print(acc/len(ans_key))
