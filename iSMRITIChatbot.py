#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:01:37 2019

@author: aviral
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:24:46 2019

@author: aviral
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:13:30 2019

@author: aviral
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:37:07 2019

@author: aviral
"""

import numpy as np
import tensorflow as tf
import re
import time
 
 
 
########## PART 1 - DATA PREPROCESSING ##########
 
 
#OPENING SYNTHETIC FILES


questions = open('/Users/aviral/Desktop/questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
answers= open('/Users/aviral/Desktop/answers.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
 

 


 
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
 
# Cleaning the questions
    

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
clean_questions=clean_questions[:621]
 
# Cleaning the answers


clean_answers = answers

clean_answers=clean_answers[:621]
clean_answersa=np.array(clean_answers)



#CREATING WORD DICTIONARY



#CREATING BAG OF WORDS MODEL


t={}
j=0
for i in np.unique(clean_answers):
   t[i]=j
   j=j+1
l=0
for k in clean_answers:
    clean_answers[l]=t[k]
    l=l+1
from keras.utils import to_categorical
Y=to_categorical(np.array(clean_answers))
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
threshold_questions = 0
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
print(max(questionswords2int.values()))
X=[]
print(len(Y))
for i in clean_questions:
    x=[]
    for h in i.split():
        x.append(int(questionswords2int[h]))
    X.append(x)
X=np.array(X)


#TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1)


#Multi hot encoding the questions


def multi_hot_encode(sequences, dimension):
 
  result=[]
  for i in sequences:
    g=[0]*dimension
    for h in i:
      g[h]=1
    result.append(g)
  result=np.array(result)
  return result
x_train = multi_hot_encode(X_train, len(questionswords2int)+1)
x_test = multi_hot_encode(X_test, len(questionswords2int)+1)


#KERAS NN MODEL


import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(199,input_dim=x_train.shape[1],activation='sigmoid'))
model.add(Dense(380,input_dim=199,activation='sigmoid'))

model.add(Dense(64,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#COMPILED THE MODEL


def fit(model):
    
    history=model.fit(x_train, Y_train, epochs=1000, batch_size=100, verbose=1)
    history_dict=history.history
    scores=model.evaluate(x_test,Y_test,verbose=1)
    return scores,history_dict
    
    
scores,history_dict = fit(model)   
Accuracy=scores[1]*100
print('Accuracy of your model is')
print(scores[1]*100) 


#CREATING FILE FOR UN-ANSWERED QUESTIONS


f= open("unansweredquestions.txt","a")
print("Enter bye to exit")


#PREDICTING THE ANSWERS


while(True):
  try:
      
    question = input("You: ")
    
    if question == 'bye':
        break
    ques=clean_text(question)
    ques=ques.split()
    y=0
    for h in ques:
        if h not in questionswords2int:
            f.write("\r\n")
            f.write(question)
            ques[y]="_"
        y=y+1
    id_to_word = {value:key for key,value in questionswords2int.items()}
    id_to_ans = {value:key for key,value in t.items()}
    train=[]
    for i in ques:
       train.append(questionswords2int[i])
    test=np.zeros(x_test.shape[1])
    for i in train:
       test[i]=1
    test=np.array([test])
    
    #CREATING HIGHEST PROBABILITY ANSWER
    
    s=sorted((model.predict(test)[0]))
    b=np.array(s[63])
    c = np.setdiff1d(s,b)
    
    max2=max(c)
    
    print(max((model.predict(test)[0])))
    print(max2)
    if (max((model.predict(test)[0]))<0.50):
        
        #CHECKING IF PROBABILITY IS TOO LOW TO BE SURE
        
        print("Sorry,I could not understand you")
        continue
    elif (max((model.predict(test)[0]))-max2<0.18):
        
        #CHECKING IF ANOTHER ANSWER HAS A HIGH PROBABILITY
        
        print("You are confusing me")
        continue
        
    answer=id_to_ans[np.argmax(model.predict(test))]
    print('ChatBot: ' + answer)
  except:
     f.write("\r\n")
     f.write(question)
     print("I: Can you please ask something relevent")
     continue
f.close()


#GUI For


 
def tkinput(ques):
 
  try:
      
    question = ques
    
    if question == 'bye' or question=='shutup' or question=='get lost' or question=='get out' or question=='getout' or question=='getlost':
        window.destroy()
        return None
    ques=clean_text(question)
    ques=ques.split()
    y=0
    for h in ques:
        if h not in questionswords2int:
            
            ques[y]="_"
        y=y+1
    id_to_word = {value:key for key,value in questionswords2int.items()}
    id_to_ans = {value:key for key,value in t.items()}
    train=[]
    for i in ques:
       train.append(questionswords2int[i])
    test=np.zeros(x_test.shape[1])
    for i in train:
       test[i]=1
    test=np.array([test])
    s=sorted((model.predict(test)[0]))
    b=np.array(s[63])
    c = np.setdiff1d(s,b)
    
    max2=max(c)
    
    print(max((model.predict(test)[0])))
    print(max2)
    if (max((model.predict(test)[0]))<0.50):
        print("Sorry,I could not understand you")
        answer="Sorry,I could not understand you"
        v=[answer,str(s[63])]
        return v
        
    elif (max((model.predict(test)[0]))-max2<0.18):
        answer="You are confusing me"
        v=[answer,str(s[63])]
        return v
        
    answer=id_to_ans[np.argmax(model.predict(test))]
    print('ChatBot: ' + answer)
    answer=('ChatBot: ' + answer)
    
    v=[answer,str(s[63])]
    return v
  except:
     
    print("I: Can you please ask something relevent")
    answer="I: Can you please ask something relevent"
    v=[answer,"0"]
    return v
     
  
    

from tkinter import *

window = Tk()
 
window.title("Welcome to iSMRITI Chatbot")
 
window.geometry('913x450')
 
lbl = Label(window, text="", padx=5, pady=5,width=100,height=5)
lbln1 = Label(window, text="", padx=5, pady=5,width=100,height=5)
lbl.grid(column=1, row=0)
lbl1=Label(window, text="", padx=5, pady=5,width=100,height=5)
lbl.grid(column=1, row=5)
lbl1.grid(column=1, row=6)
txt = Entry(window,width=70)

txt.grid(column=1, row=1)
lbln1 = Label(window, text="", padx=5, pady=5,width=100,height=5)
lbl.grid(column=1, row=2)
lbln1 = Label(window, text="", padx=5, pady=5,width=100,height=5)
lbl.grid(column=1, row=4)

def clicked():
 
    lbl.configure(text=tkinput(txt.get())[0])
    lbl1.configure(text="Answer's Probability:"+tkinput(txt.get())[1])

    
btn = Button(window, text="Get An Answer", command=clicked,width=100,height=5)
 
btn.grid(column=1, row=3)
 
window.mainloop()
    
