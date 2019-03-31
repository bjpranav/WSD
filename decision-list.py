# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:40:00 2019

@author: Pranav Krishna
"""
#Importing required packages
import xml.etree.ElementTree as ET
import re
import nltk

#Fetches the xml file and stores it in variable called tree
tree = ET.parse(r'D:\bin\AIT-690\Assignments\wsd\PA3\PA3\line-train.xml')
#Points to the root of the tree
root = tree.getroot()
line=root.find('lexelt')

#Returns two words before and after the ambigus word
def bigram(xmlstr):
    features=re.findall('(\w+) (\w+) <head>\w{4,5}<\/head> (\w+) (\w+)', str(xmlstr), re.IGNORECASE)
    return(features)

#Returns a word before and after the ambigus word
def unigram(xmlstr):
    features=re.findall('(\w+) <head>\w{4,5}<\/head> (\w+)', str(xmlstr), re.IGNORECASE)
    return(features)
#Returns a word before the ambigus word
def kminus1(xmlstr):
    features=re.findall('(\w+) <head>', str(xmlstr), re.IGNORECASE)
    return(features)
#Returns a word after the ambigus word
def kplus1(xmlstr):
    features=re.findall('</head> (\w+)', str(xmlstr), re.IGNORECASE)
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
    
#Iterates through all the instances
for val in line:
    #Stores the instance ID
    instanceId=(val.attrib)['id']
    #Iterates through all the tags under instance tag
    for context in val:
        #If the tag has sense id the that is stored in the senseId variable
        if((context.attrib)!={}):
            senseId=(context.attrib)['senseid']
        #If the tag contains the string and the ambiguous word, fowwing are performed
        else:
            #Converts the tags into strings
            xmlstr = ET.tostring(context)
            #Gets the word before the ambigus word
            beforeWord=kminus1(xmlstr)
            #Gets the word after the ambigus word
            afterWord=kplus1(xmlstr)
            #Gets a word before and after the ambigus word
            unigramFeature=unigram(xmlstr)
            #Gets two words before and after the ambigus word
            bigramFeature=unigram(xmlstr)
            # Gets the tag before and after the ambiguous tag
            tags=tagger(xmlstr)
            beforeTag=tags[0]
            afterTag=tags[1]
            print(beforeTag,afterTag)

