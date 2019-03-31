
#Importing required packages
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
import nltk

#Fetches the xml file and stores it in variable called tree
tree = ET.parse(r'D:\bin\AIT-690\Assignments\wsd\PA3\PA3\line-train.xml')
#Points to the root of the tree
root = tree.getroot()
line=root.find('lexelt')

#Returns two words before and after the ambigus word
def bigram(xmlstr):
    features=re.findall('(\w+ \w+)"?.?-? <head>\w{4,5}<\/head> (\w+ \w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return('EOL')
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
    else:
        print(cleanedWords)
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
            features.append('EOL')
    return(features)

#Returns a word before and after the ambigus word
def unigram(xmlstr):
    features=re.findall('(\w+)"?.?-? <head>\w{4,5}<\/head> (\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return(['EOL'])
    else:
        return(features)
    return(features)
#Returns a word before the ambigus word
def kminus1(xmlstr):
    features=re.findall('"?(\w+)"?.?-? <head>', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return(['EOL'])
    else:
        return(features)
    return(features)
#Returns a word after the ambigus word
def kplus1(xmlstr):
    features=re.findall(r'</head> [,.;"]? ?(\w+)', str(xmlstr), re.IGNORECASE)
    if(features==[]):
        return(['EOL'])
    else:
        return(features)

#Tags POS for all the words and returns the tag of the words before and after the 
#ambiguous word
def tagger(xmlstr):
    #features=re.findall('</head> (\w+)', str(xmlstr), re.IGNORECASE)
    index=None
    querywords = str(xmlstr).replace('</s>','').replace('<s>','').replace('</context>','').replace('\n','').replace('<context>','').replace('\\n\\n','').replace('\\n','')     
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
#Iterates through all the instances
for val in line:
    #Stores the instance ID
    instanceId=(val.attrib)['id']
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
            print(kwordsBefore(xmlstr,3))


def setz(sequence):
    sequenceList=[]
    for i in sequence:
        if(i[0] not in sequenceList):
            sequenceList.append(i[0])
    return(sequenceList)



def initialize(sequence1,sequence2):
    return(np.zeros((len(sequence1), len(sequence2))))
    
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
