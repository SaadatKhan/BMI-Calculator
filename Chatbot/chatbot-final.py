import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import re 
import time 
tf.compat.v1.disable_eager_execution()

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
    inputs = tf.compat.v1.placeholder(tf.int32, [None,None], name = 'input')
    targets = tf.compat.v1.placeholder(tf.int32, [None,None], name = 'targets')
    lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name = 'keep_prob')
    return(inputs,targets,lr,keep_prob)

#Making Function for targets batches:
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0], [batch_size,-1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    preprocessed_targets
    
# Creating Encoding RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers,keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype=tf.float32)
    return encoder_state

#Decoding the training set


def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_opttion = 'bahdanau' , num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_fn_train(encoder_state[0],
                                                                      attention_keys,
                                                                      attention_values,
                                                                      attention_score_function,
                                                                      attention_construct_function,
                                                                      name = 'attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    

# Decoding the testing set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix,sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_opttion = 'bahdanau' , num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_fn_inference(output_function,
                                                                      encoder_state[0],
                                                                      attention_keys,
                                                                      attention_values,
                                                                      attention_score_function,
                                                                      attention_construct_function,
                                                                      decoder_embeddings_matrix,
                                                                      sos_id,
                                                                      eos_id,
                                                                      maximum_length,
                                                                      num_words,
                                                                      name = 'attn_dec_inf')
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              test_decoder_function,
                                                                                                              scope = decoding_scope)
   
    return test_predictions

#Creating Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size ):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x:tf.contrib.layers.fully_connected(x,
                                                                     num_words,
                                                                     None,
                                                                     scope = decoding_scope,
                                                                     weights_initializers= weights,
                                                                     biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions
        
#Building the seq2seq model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1 , decoder_embedding_size], 0,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                          decoder_embeddings_matrix,
                                                          encoder_state,
                                                          questions_num_words,
                                                          sequence_length,
                                                          rnn_size,
                                                          num_layers,
                                                          questionswords2int,
                                                          keep_prob,
                                                          batch_size)
    return training_predictions, test_predictions
    
#Setting the Hyperparameters 

epochs =100 
batch_size = 64
rnn_size = 512
num_layers =3
encoding_embedding_size = 512      
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session 
tf.compat.v1.reset_default_graph()
session=tf.compat.v1.InteractiveSession()

#Loading Model 

inputs, targets, lr, keep_prob = model_inputs()

#Setting the sequence length
sequence_length = tf.compat.v1.placeholder_with_default(25, None, name = 'sequence_length')

input_shape = tf.shape(inputs)

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)

#setting up the loss error, the optimizer and gradient clipping

with tf.name_scope("Optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets,
                                                  tf.ones(input_shape[0],sequence_length))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.computer_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_balue(grad_tensor, -5.,5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

def apply_padding(batch_of_sequences, word2int ):
    max_sequence_length = max([len(sequence)for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']]* (max_sequence_length - len(sequence))for sequence in batch_of_sequences]

def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index*batch_size
        questions_in_batch = questions[start_index: start_index+batch_size]
        answers_in_batch = answers[start_index: start_index+batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
training_validation_split = int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
        
#Training

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions))//batch_size//2)- 1   
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer()) 
for epoch in range(1,epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs:padded_questions_in_batch,
                                                                                               targets:padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error+=batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time-starting_time
        if batch_index%batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f},Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                      epochs,
                                                                                                                                      batch_index,
                                                                                                                                      len(training_questions)//batch_size,
                                                                                                                                      total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                      int(batch_time*batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss ==0 and batch_index>0:
            total_validation_loss_error =0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs:padded_questions_in_batch,
                                                                       targets:padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error+=batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time-starting_time
            average_validaton_loss_error = total_validation_loss_error/ (len(validation_questions)/ batch_size)
            print('Validaton Loss Error: {:0.63f}, Batch Validation Time: {:d} seconds'.format(average_validaton_loss_error, int(batch_time)))
            
            learning_rate *= learning_rate_decay
            if learning_rate< min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validaton_loss_error)
            if average_validaton_loss_error<=min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver=tf.train.Saver()
                saver.save(session, checkpoint)
            else: 
                print("Sorry I dont speak better, I need more training")
                early_stopping_check+=1
                if early_stopping_check == early_stopping_stop:
                    break
        if early_stopping_check == early_stopping_stop:
            print("I have reached my best")
            break
print('Game Over')
            
            
            
                
            
  
        
        
    