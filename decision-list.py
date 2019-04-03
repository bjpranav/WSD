
#Importing required packages
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
import nltk
import random
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
import sys
from nltk.corpus import stopwords
from collections import Counter


path=sys.argv[1]
tree=ET.parse(path)

path2=sys.argv[2]
tree1=ET.parse(path2)

decision_list=sys.argv[3]


random.seed(12345)

#Fetches the xml file and stores it in variable called tree
#tree = ET.parse(r'D:\bin\AIT-690\Assignments\wsd\PA3\PA3\line-train.xml')

#tree = ET.parse(r'C:\Users\alaga\Desktop\sem 2\AIT690\WSD\line-train.xml')
#Points to the root of the tree
root = tree.getroot()
line=root.find('lexelt')

#Returns two words before and after the ambigus word
def bigram(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('(\w+ \w+)"?.?-? <head>\w{4,5}<\/head> (\w+ \w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)

def kwordsBefore(xmlstr,k):
    features=[]
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(".", " ")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    cleanedWords = xmlstr.replace(",", "")
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
    xmlstr=str(xmlstr).lower()
    features=[]
    xmlstr = xmlstr.replace(",", "")
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(".", " ")
    cleanedWords = xmlstr.replace(",", "")
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
        if(len(bag_of_words)>index+i):
            features.append(bag_of_words[index+i])
        else:
            features.append('E O L')
    return(features)

#Returns a word before and after the ambigus word
def unigram(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('(\w+)"?.?-? <head>\w{4,5}<\/head> (\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word before the ambigus word
def kminus1(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
    features=re.findall('"?(\w+)"?.?-? <head>', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('E O L')
    else:
        return(features)
    return(features)
#Returns a word after the ambigus word
def kplus1(xmlstr):
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
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
    xmlstr=str(xmlstr).lower()
    xmlstr = xmlstr.replace("<s>","")
    xmlstr = xmlstr.replace("</s>", "")
    xmlstr = xmlstr.replace(r'\n', "")
    xmlstr = xmlstr.replace("<context>", "")
    xmlstr = xmlstr.replace("</context>", "")
    xmlstr = xmlstr.replace(",", "")
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
productList=''
senseList=''
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
            if(senseId=='product'):
                productList+=(str(xmlstr).lower())
            else:
                senseList+=(str(xmlstr).lower())
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

productList=str(productList.lower())
productList = productList.replace("<s>","")
productList = productList.replace("</s>", "")
productList = productList.replace(r'\n', "")
productList = productList.replace("<context>", "")
productList = productList.replace("</context>", "")

senseList=str(senseList.lower())
senseList = senseList.replace("<s>","")
senseList = senseList.replace("</s>", "")
senseList = senseList.replace(r'\n', "")
senseList = senseList.replace("<context>", "")
senseList = senseList.replace("</context>", "")

stopWords=set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\w+')

cleanedWordsProductList=tokenizer.tokenize(productList)
cleanedWordsProductList=[word for word in cleanedWordsProductList if not word in stopWords]

cleanedWordsPhoneList=tokenizer.tokenize(senseList)
cleanedWordsPhoneList=[word for word in cleanedWordsPhoneList if not word in stopWords]

intersection = Counter(cleanedWordsProductList) & Counter(cleanedWordsPhoneList)
ProductListCounts=Counter(cleanedWordsProductList)-intersection
PhoneListCounts=Counter(cleanedWordsPhoneList)-intersection
PhoneListCounts.most_common(305)

#list(nltk.FreqDist(cleanedWords))

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
            if (mer in kplus2words.index):
                product = kplus2words.at[mer, 'product']
                phone = kplus2words.at[mer, 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 3):
            mer = temp_list[0] + " " + temp_list[1]
            if(mer in kminus2words.index):
                product = kminus2words.at[mer, 'product']
                phone = kminus2words.at[mer, 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 4):
            if(temp_list[0] in oneWordAfter.index):
                product = oneWordAfter.at[temp_list[0], 'product']
                phone = oneWordAfter.at[temp_list[0], 'phone']

            else:
                product = 0
                phone = 0

        elif (flag == 5):
            if(temp_list[0] in oneWordBefore.index):
                product = oneWordBefore.at[temp_list[0], 'product']
                phone = oneWordBefore.at[temp_list[0], 'phone']

            else:
                product = 0
                phone = 0
            
        elif (flag == 6):
            if(temp_list[0] in beforeTagDf.index):
                product = beforeTagDf.at[temp_list[0], 'product']
                phone = beforeTagDf.at[temp_list[0], 'phone']
                #print(6)
            else:
                product = 0
                phone = 0
            
        elif (flag == 7):
            if(temp_list[0] in afterTagDf.index):
                product = afterTagDf.at[temp_list[0], 'product']
                phone = afterTagDf.at[temp_list[0], 'phone']
                #print(7)
            else:
                product = 0
                phone =    0

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


#tree = ET.parse(r'D:\bin\AIT-690\Assignments\wsd\PA3\PA3\line-test.xml')
#tree1 = ET.parse(r'C:\Users\alaga\Desktop\sem 2\AIT690\WSD\line-test.xml')
#Points to the root of the tree
root = tree1.getroot()
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

jj=None
count=0
bi_prob=[]
uni_prob=[]
minus2_prob=[]
plus2_prob=[]
minus1_prob=[]
plus1_prob=[]
beforeTag_prob=[]
afterTag_prob=[]
final=[]
collectionList=[]
x=-1
for i in content:
    jj=None
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
    #print(g)
    
    h=[tags[1]]
    flag=7
    prob,hh = prob_finder(h, flag)
    afterTag_prob.append(prob)
    
    tokenizedString=nltk.word_tokenize(i.lower())
    productSet=['company']
    phoneSet=['telephone']
    

    if('consumer' in tokenizedString):
        jj='product'
    
    elif('phone' in tokenizedString):
        jj='phone'
       
    elif('telephone' in tokenizedString):
        jj='phone'
        
    elif('call' in tokenizedString):
        jj='phone'

    elif('calls' in tokenizedString):
        jj='phone'
    
    
    if(len(set(tokenizedString)&set(productSet))>0):
        jj='product'
        #print(set(tokenizedString)&set(productSet))
    elif(len(set(tokenizedString)&set(phoneSet))>0):
        jj='phone'
        #print(set(tokenizedString)&set(phoneSet))

    collection=[bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]]
    collectionList.append(collection)
    reorder=np.argmax([bi_prob[x],uni_prob[x],plus2_prob[x],minus2_prob[x],plus1_prob[x],minus1_prob[x],beforeTag_prob[x],afterTag_prob[x]])
    
    if(jj):
        final.append(jj)        
    elif(reorder==7):
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


col=['Bigram','Unigram','Plus2_Words','Minus2_Words','Plus1_Word','Minus1_Word','Before_POS_Tag','After_POS_Tag']
decision_list_df=pd.DataFrame()

for i in range(0,len(collectionList)):
    for j in range(0,len(col)):
        decision_list_df.loc[i, col[j]]=collectionList[i][j]

decision_list_df['Sense']=final


with open('my-decision-list.txt', 'w') as f:
    f.write(decision_list_df.to_string())

my_list=""
for i in range(0,len(instanceId)):
    temp='<answer instance="'+instanceId[i]+'" senseid="'+final[i]+'"/>'
    my_list=my_list+temp+'\n'

print(my_list)
