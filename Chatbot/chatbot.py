import numpy as np
import pandas as pd
import tensorflow as tf
import re 
import time 

lines = open('movie_lines.txt', encoding='utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors = 'ignore').read().split('\n')


id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]= _line[4]

# list for all the conversations    
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
    
#getting separately the questions and answers

questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


def text_cleaner(text):
    
    text= text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"he's","he is", text) 
    text = re.sub(r"she's","she is", text)
    text = re.sub(r"that's","that is", text)
    text = re.sub(r"you're","you are", text)
    text = re.sub(r"what's","what is", text)
    text = re.sub(r"where's ","where is", text)
    text = re.sub(r"\'ll"," will", text)
    text = re.sub(r"\'ve"," have", text)
    text = re.sub(r"\'re"," are", text)
    text = re.sub(r"\'d"," would", text)
    text = re.sub(r"won't","will not", text)
    text = re.sub(r"can't","cannot", text)
    text = re.sub("[-,(,),\",#,/,@,;,:,<,>,{,},+,=,-,|,.,?]","", text)
   
    return text
    
cleaned_q = []

for question in questions:
    cleaned_q.append(text_cleaner(question))
    
cleaned_a = []
for answer in answers:
    cleaned_a.append(text_cleaner(answer))
    
# Creating Word Frequency
word2count= {}

for question in cleaned_q:
    for word in question.split():
        if word in word2count.keys():
            word2count[word]+=1
        else:
            word2count[word]=1
for answer in cleaned_a:
    for word in answer.split():
        if word in word2count.keys():
            word2count[word]+=1
        else:
            word2count[word]=1
        
#create two dictionaries that map the questions words and the answers words to a unique integer
threshold = 20
questionswords2int = {}
word_number = 0 

for word, count in word2count.items():
    if count>= threshold:
        questionswords2int[word]= word_number
        word_number+=1

answerswords2int = {}
word_number = 0 

for word, count in word2count.items():
    if count>= threshold:
        answerswords2int[word]= word_number
        word_number+=1
                
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token]= len(questionswords2int)+1

for token in tokens:
    answerswords2int[token]= len(answerswords2int)+1

answersint2word= {w_i: w for w, w_i in answerswords2int.items()}

for i in range (len(cleaned_a)):
    cleaned_a[i]+= ' <EOS>'


# Making Questions and Answers into respective integers
questions_to_int = []
for question in cleaned_q:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)
answers_to_int = []
for answer in cleaned_a:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
    
# Sorted Cleaned Questions an answers based on the length of questions and answers

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
       
        
       

# Creating placehorlders for the iinputs and the targets

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.placeholder(tf.int32, [None,None], name = 'targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return(inputs,targets,lr,keep_prob)

#Making Function for targets batches:
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0], [batch_size,-1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    preprocessed_targets
    
