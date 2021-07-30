from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from utils import clean_text
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences



DATA = input("enter name of processed data file: ")
MODEL_NAME = DATA
data = pickle.load(open(f"./processed_data/{DATA}.pickle", "rb"))
decoder_final_output = data["decoder_final_output"]
vocab = data["vocab"]
encoder_input = data["encoder_input"]
decoder_input = data["decoder_input"]
inv_vocab = data["inv_vocab"]


decoder_final_output = to_categorical(decoder_final_output, len(vocab))



enc_inp = Input(shape=(13, ))
dec_inp = Input(shape=(13, ))


VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE + 1, output_dim= 50, input_length= 13, trainable= True)

enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)


model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

EPOCHS = int(input("enter the number of epochs: "))
model.fit([encoder_input, decoder_input],decoder_final_output,epochs=EPOCHS)

model.save(f"./models/{MODEL_NAME}/model/")


enc_model = Model([enc_inp], enc_states)

decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = dec_lstm(dec_embed, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

dec_model = Model([dec_inp]+ decoder_states_inputs,[decoder_outputs]+ decoder_states)


enc_model.save(f"./models/{MODEL_NAME}/enc_model/")
dec_model.save(f"./models/{MODEL_NAME}/dec_model/")

pickle_out = open(f"./models/{MODEL_NAME}/dense.pickle","wb")
pickle.dump(dense, pickle_out)
pickle_out.close()



