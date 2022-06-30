from __future__ import print_function, division
from builtins import range, input

import os
import sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass

BATCH_SIZE = 64  # batch size for the training
EPOCHS = 40  # Number of epochs to train for
LATENT_DIM = 256  # Latend dimensionality of the encoding space.
NUM_SAMPLES = 10000  # Number of samples to train on
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# where we will store the data
input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
target_texts_inputs = []  # sentence in target language offset by 1

# load in the data
t = 0
for line in open('C:\\Users\\manpreet singh\\Downloads\\languagetranslationdataset\\spa.txt', encoding='utf-8'):
    # only keep a limited number of samples
    t += 1
    if t > NUM_SAMPLES:
        break

    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation = line.split('\t')

    # make the target input and output
    # recall we'll be using teacher forcing
    target_text = translation + ' <eos>'
    # for teacher forcing in the decoder side
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)
print("num samples:", len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping of the input language
word2index_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2index_inputs))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2index_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2index_outputs))
print("djfoijfsoiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
count = 0
for k,v in word2index_outputs.items():
    if count < 10:
        print(f"key : {k}    value : {v}")
        count += 1
    else : 
        break
    
# store number of output words for later
# adding 1 since indexing starts at 1
num_words_output = len(word2index_outputs) + 1

# determine maximum length output sequences
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_data.shape :", encoder_inputs.shape)
print("encoder_data[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_data[0]:", decoder_inputs[0])
print("decoder_data.shape", decoder_inputs.shape)

decoder_targets = pad_sequences(
    target_sequences, maxlen=max_len_target, padding='post')


# store all the pre-trained word vectors
print("Loading word vectors...")
word2vec = {}
with open(os.path.join('C:\\Users\\manpreet singh\\Downloads\\glove.6B\\glove.6B.%sd.txt' % EMBEDDING_DIM), encoding="utf8") as glove_file:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2]
    for line in glove_file:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2index_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2index_inputs.items():
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector


# create embedding layer
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input,
    # trainable=True
)

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_one_hot = np.zeros(
    (len(input_texts), max_len_target, num_words_output), dtype='float32')
print(f"one hot shape is  : {decoder_targets_one_hot.shape}")
# assign the values to one hot
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        # here i is the layer or the input sample and in that layer there is a matrix in which rows:padded number of words(8) and columns:total unique words in the dataset so if in a sentence(layer) the word present at 0 position will correspond to t and  1 will be marked where the word is present corresponding to the columns
        decoder_targets_one_hot[i, t, word] = 1

# build the model

encoder_inputs_placeholder = Input(shape=(max_len_input,))
encoder_inputs_x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)# latent dimension is also called lstm nodes
encoder_outputs, h, c = encoder(encoder_inputs_x)
# encoder_outputs, h = encoder(encoder_inputs_x) #gru

# keep only the states to pass into decoder i.e. the last hidden
encoder_states = [h, c]
# encoder_states = [state_h] # gru

# set up the decoder, using [h,c] as initial state.
decoder_inputs_placeholder = Input(shape=(max_len_target,))

# this word embedding will not use pre-trained vectors
# although you could
decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have
# return_sequences = True
decoder_lstm = LSTM( # make the lstm
    LATENT_DIM, # lstm nodes
    return_sequences=True,
    return_state=True, 
    dropout=0.5
    )
decoder_outputs, _, _ = decoder_lstm( # giving input to the lstm
    decoder_inputs_x, 
    initial_state=encoder_states
    )

# decoder_outputs, _ = decoder_gru(decoder_inputs_x, initial_state=encoder_state)

# final dense layer for prediction
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs) #decoder output after passing it through the dense layer

# create the model object
model = Model([encoder_inputs_placeholder,
              decoder_inputs_placeholder], decoder_outputs)




# compile the model and train it
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracy
# plt.plot(r.history['acc'], label='acc')
# plt.plot(r.history['val_acc'], label='val_acc')
# plt.legend()
# plt.show()


# Save model
model.save('seq2seqmachinetranslation.h5')

# Make predictions
# As with the poetry example, we need to create another model
# that can take in the RNN state and previous word as input
# and accept a T = 1 sequence

# The encoder will be stand-alone
# From this we will get our initial decoder hidden state

# here encoder_inputs_placeholder is the input of the model and the output is encoder_states and these encoder_states will be later passed on to the decoder as the initial input
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_states_inputs = [decoder_state_inputs_h] for gru

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# this time, we want to keep the states too, to be output
# by our sampling model

decoder_outputs, h, c = decoder_lstm(
    decoder_inputs_single_x,
    initial_state=decoder_states_inputs
)

# decoder_outputs, state_h= decoder_lstm(
#     decoder_inputs_single_x,
#     initial_state=decoder_states_inputs
# ) for gru

decoder_states = [h, c]
# decoder_states=[h]
decoder_outputs = decoder_dense(decoder_outputs)

# The sampling model
# inputs : y(t - 1), h(t - 1), c(t - 1)
# outputs : y(t), h(t), c(t)
decoder_model = Model(
    # in a model there goes an input and an output
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# map indexes back into real words
# so we can get the results in readable form
index2word_eng = {v:k for k,v in word2index_inputs.items()}
index2word_trans = {v:k for k,v in word2index_outputs.items()}

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_values = encoder_model.predict(input_seq) # pass the input sequences to the encoder_model which predicts the hidden state and the cell state, which are sorted in states_values

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first character of target sequence with the '<sos>'
    # NOTE: tokenizer lower-cases all words
    target_seq[0,0] = word2index_outputs['<sos>']# we define a variable target_seq, which is a 1 x 1 matrix of all zeros. The target_weq variable contains the first word to the decoder model, which is <sos>

    # if we get this we break
    eos = word2index_outputs['<eos>'] # stores index value of eos

    # Create the translation
    output_sentence = [] # contains the predicted translation
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_values # in the argument we put target seq which was the previous word and states_values which is hidden state and cell state
        )
        # output_tokens, h = decoder_model.predict(
        #       [target_seq] + states_values 
        # ) for gru
        predindex = np.argmax(output_tokens[0,0,:]) # np.argmax predicts the most probable word and hence we pass 0 indexed(most probable) therefore idx stores the most probable or predicted word
        if eos == predindex:
            break
        word = ''
        if predindex > 0:
            word = index2word_trans[predindex]
            output_sentence.append(word)
        
        target_seq[0,0] = predindex
        states_values = [h, c] 
    return ' '.join(output_sentence)

while True:
    # Do some test translations
    i = np.random.choice(len(input_texts))# choose a random word index from input texts
    input_sequence = encoder_inputs[i:i+1]
    translation = decode_sequence(input_sequence)
    print('-')
    print('Input : ', input_texts[i])
    print('Translation : ', translation)

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break

