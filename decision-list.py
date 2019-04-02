
#Importing required packages
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
import nltk
import random

#Fetches the xml file and stores it in variable called tree
#tree = ET.parse(r'D:\bin\AIT-690\Assignments\wsd\PA3\PA3\line-train.xml')

tree = ET.parse(r'C:\Users\alaga\Desktop\sem 2\AIT690\WSD\line-train.xml')
#Points to the root of the tree
root = tree.getroot()
line=root.find('lexelt')

#Returns two words before and after the ambigus word
def bigram(xmlstr):
    features=re.findall('(\w+ \w+)"?.?-? <head>\w{4,5}<\/head> (\w+ \w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)

def kwordsBefore(xmlstr,k):
    features=[]
    cleanedWords = str(xmlstr).replace('</s>','').replace('<s>','').replace('</context>','').replace('\n','').replace('<context>','').replace('\\n\\n','').replace('\\n','').replace('.',' ')     
    bag_of_words=cleanedWords.split()
    
    if('<head>line</head>' in cleanedWords):
        index=bag_of_words.index('<head>line</head>')
    elif('<head>lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>lines</head>')
    elif('<head>Lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>Lines</head>')
    elif('<head>Line</head>' in bag_of_words):
        index=bag_of_words.index('<head>Line</head>')
    for i in range(1,k+1):
        features.append(bag_of_words[index-i])
    return(features)
    
def kwordsAfter(xmlstr,k):
    features=[]
    cleanedWords = str(xmlstr).replace('</s>','').replace('<s>','').replace('</context>','').replace('\n','').replace('<context>','').replace('\\n\\n','').replace('\\n','').replace('.',' ')     
    bag_of_words=cleanedWords.split()
    
    if('<head>line</head>' in cleanedWords):
        index=bag_of_words.index('<head>line</head>')
    elif('<head>lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>lines</head>')
    elif('<head>Lines</head>' in bag_of_words):
        index=bag_of_words.index('<head>Lines</head>')
    elif('<head>Line</head>' in bag_of_words):
        index=bag_of_words.index('<head>Line</head>')
    else:
        print(cleanedWords)
    for i in range(1,k+1):
        if(len(bag_of_words)>index+i):
            features.append(bag_of_words[index+i])
        else:
            features.append('E O L')
    return(features)

#Returns a word before and after the ambigus word
def unigram(xmlstr):
    xmlstr=str(xmlstr)
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    features=re.findall('(\w+)"?.?-? <head>\w{4,5}<\/head> (\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word before the ambigus word
def kminus1(xmlstr):
    xmlstr=str(xmlstr)
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    features=re.findall('"?(\w+)"?.?-? <head>', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word after the ambigus word
def kplus1(xmlstr):
    xmlstr=str(xmlstr)
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    features=re.findall(r'</head> [,.;"]? ?(\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)

#Tags POS for all the words and returns the tag of the words before and after the 
#ambiguous word
def tagger(xmlstr):
    #features=re.findall('</head> (\w+)', str(xmlstr), re.IGNORECASE)
    index=None
    xmlstr=str(xmlstr)
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    querywords=xmlstr
    #querywords = str(xmlstr).replace('</s>','').replace('<s>','').replace('</context>','').replace('\n','').replace('<context>','').replace('\\n\\n','').replace('\\n','')     
    wordsList=querywords.split()
    if('<head>line</head>' in wordsList):
        index=wordsList.index('<head>line</head>')
    elif('<head>lines</head>' in wordsList):
        index=wordsList.index('<head>lines</head>')
    textforTagging=querywords.replace('<head>','').replace('</head>','')
    store=nltk.pos_tag(textforTagging.split())  
    if(index):
        beforeTag=store[index-1][1]
        afterTag=store[index+1][1]
        return([beforeTag,afterTag])
    else:
        return(['NN','NN'])
        
        
beforeWord=[]
afterWord=[]
unigramFeature=[]
bigramFeature=[]
beforeTag=[]
afterTag=[]
senses=[]
kminus2Feature=[]
kplus2Feature=[]
instanceId=[]
#Iterates through all the instances
for val in line:
    #Stores the instance ID
    instanceId.append((val.attrib)['id'])
    #Iterates through all the tags under instance tag
    for context in val:
        #If the tag has sense id the that is stored in the senseId variable
        if((context.attrib)!={}):
            senseId=(context.attrib)['senseid']
            senses.append(senseId)
        #If the tag contains the string and the ambiguous word, fowwing are performed
        else:
            #Converts the tags into strings
            xmlstr = ET.tostring(context)
            #Gets the word before the ambigus word
            beforeWord.append(((kminus1(xmlstr))[0],senseId))
            #Gets the word after the ambigus word
            afterWord.append(((kplus1(xmlstr))[0],senseId))
            #Gets a word before and after the ambigus word
            unigramString=' '.join((unigram(xmlstr))[0])
            unigramFeature.append((unigramString,senseId))
            #Gets two words before and after the ambigus word
            bigramString=' '.join((bigram(xmlstr))[0])
            bigramFeature.append((bigramString,senseId))
            # Gets the tag before and after the ambiguous tag
            tags=tagger(xmlstr)
            beforeTag.append((tags[0],senseId))
            afterTag.append((tags[1],senseId))
            #print(beforeTag,afterTag)
            kminus2Feature.append((' '.join(kwordsBefore(xmlstr,2)),senseId))
            kplus2Feature.append((' '.join(kwordsAfter(xmlstr,2)),senseId))
            #print(kwordsBefore(xmlstr,3))


def setz(sequence):
    sequenceList=[]
    for i in sequence:
        if(i[0] not in sequenceList):
            sequenceList.append(i[0])
    return(sequenceList)



def initialize(sequence1,sequence2):
    return(np.ones((len(sequence1), len(sequence2))))
    
def dfBuilder(featureSequence):
    uniqueList=setz(featureSequence)
    initializeSet = initialize((uniqueList), set(senses))
    initializedFeature=pd.DataFrame(initializeSet, index=uniqueList, columns=set(senses))
    for i in featureSequence:
        initializedFeature.at[i[0], i[1]]=initializedFeature.at[i[0], i[1]] + 1
    return(initializedFeature)


oneWordBefore=dfBuilder(beforeWord)
oneWordAfter=dfBuilder(afterWord)
unigramwords=dfBuilder(unigramFeature)
bigramwords=dfBuilder(bigramFeature) 
kminus2words=dfBuilder(kminus2Feature)
kplus2words=dfBuilder(kplus2Feature)
beforeTagDf=dfBuilder(beforeTag)
afterTagDf=dfBuilder(afterTag)





content=[]
cnt=1
for val in line:
    for context in val:
        if(cnt%2==0):
            temp=str(ET.tostring(context))
            temp = temp.replace("<s>","")
            temp = temp.replace("</s>", "")
            temp = temp.replace(r'\n', "")
            temp = temp.replace("<context>", "")
            temp = temp.replace("</context>", "")
            content.append(temp)
        cnt+=1






def prob_finder(temp_list,flag):
    if (temp_list != 'E O L'):


        if(flag==0):

            mer = temp_list[0][0] + " " + temp_list[0][1]
            if(mer in bigramwords.index):
                product = bigramwords.at[mer, 'product']
                phone = bigramwords.at[mer, 'phone']
            else:
                product=0
                phone=0

        elif(flag==1):
            mer = temp_list[0][0] + " " + temp_list[0][1]
            if (mer in unigramwords.index):
                product = unigramwords.at[mer, 'product']
                phone = unigramwords.at[mer, 'phone']
            else:
                product = 0
                phone = 0

        elif (flag == 2):

            mer = temp_list[0] + " " + temp_list[1]
            if (mer in kplus2words):
                product = kplus2words.at[mer, 'product']
                phone = kplus2words.at[mer, 'phone']
            else:
                product = 0
                phone = 0

        elif (flag == 3):
            mer = temp_list[0] + " " + temp_list[1]
            if(mer in kminus2words):
                product = kminus2words.at[mer, 'product']
                phone = kminus2words.at[mer, 'phone']
            else:
                product = 0
                phone = 0

        elif (flag == 4):
            if(temp_list[0] in oneWordAfter):
                product = oneWordAfter.at[temp_list[0], 'product']
                phone = oneWordAfter.at[temp_list[0], 'phone']
            else:
                product = 0
                phone = 0

        elif (flag == 5):
            if(temp_list[0] in oneWordBefore):
                product = oneWordBefore.at[temp_list[0], 'product']
                phone = oneWordBefore.at[temp_list[0], 'phone']
            else:
                product = 0
                phone = 0
            
        elif (flag == 6):
            if(temp_list[0] in beforeTagDf):
                product = beforeTagDf.at[temp_list[0], 'product']
                phone = beforeTagDf.at[temp_list[0], 'phone']
            else:
                product = 0
                phone = 0
            
        elif (flag == 7):
            if(temp_list[0] in afterTagDf):
                product = afterTagDf.at[temp_list[0], 'product']
                phone = afterTagDf.at[temp_list[0], 'phone']
            else:
                product = 0
                phone = 0

        if (product == phone):
            prob = max(product, phone)
            wsd=random.choice(["product","phone"])

        else:
            prob = np.log(product / phone)
            if(product>phone):
                wsd="product";
            else:
                wsd="phone"


    else:
        prob = 0
        wsd = random.choice(["product", "phone"])


    return abs(prob),wsd


bi_prob=[]
uni_prob=[]
minus2_prob=[]
plus2_prob=[]
minus1_prob=[]
plus1_prob=[]
beforeTag_prob=[]
afterTag_prob=[]
final=[]
x=-1
for i in content:
    x+=1
    a = bigram(i)
    flag=0
    prob,aa=prob_finder(a,flag)
    bi_prob.append(prob)

    b = unigram(i)
    flag=1
    prob,bb = prob_finder(b,flag)
    uni_prob.append(prob)

    c = kwordsAfter(i, 2)
    flag = 2
    prob,cc = prob_finder(c, flag)
    plus2_prob.append(prob)

    d =kwordsBefore(i,2)
    flag = 3
    prob,dd = prob_finder(d, flag)
    minus2_prob.append(prob)

    
    e = kplus1(i)
    flag = 4 
    prob,ee = prob_finder(e, flag)
    plus1_prob.append(prob)
    

    f = kminus1(i)
    flag = 5
    prob,ff = prob_finder(f, flag)
    minus1_prob.append(prob)
    
    tags=tagger(i)
    g=[tags[0]]
    flag=6
    prob,gg = prob_finder(g, flag)
    beforeTag_prob.append(prob)
    
    h=[tags[1]]
    flag=7
    prob,hh = prob_finder(h, flag)
    afterTag_prob.append(prob)


    collection=[bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]]
    reorder=np.argmax([bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]])

    if(reorder==7):
        final.append(hh)
    elif(reorder==6):
        final.append(gg)
    elif(reorder==5):
        final.append(ff)
    elif(reorder==4):
        final.append(ee)
    elif (reorder == 3):
        final.append(dd)
    elif (reorder == 2):
        final.append(cc)
    elif (reorder == 1):
        final.append(bb)
    elif (reorder == 0):
        final.append(aa)



tree = ET.parse(r'C:\Users\alaga\Desktop\sem 2\AIT690\WSD\line-test.xml')
#Points to the root of the tree
root = tree.getroot()
line1=root.find('lexelt')

content=[]
instanceId=[]
for val in line1:
    instanceId.append((val.attrib)['id'])
    for context in val:
        temp=str(ET.tostring(context))
        temp = temp.replace("<s>","")
        temp = temp.replace("</s>", "")
        temp = temp.replace(r'\n', "")
        temp = temp.replace("<context>", "")
        temp = temp.replace("</context>", "")
        content.append(temp)




key=open(r"C:\Users\alaga\Desktop\sem 2\AIT690\WSD1\line-answers.txt")
key=key.read()


my_list=""
for i in range(0,len(instanceId)):
    temp='<answer instance="'+instanceId[i]+'" senseid="'+final[i]+'"/>'
    my_list=my_list+temp+'\n'



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

