
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from utils import clean_text

lines_file = input("enter name of lines file (include extension): ")
convos_file = input("enter name of convos file (include extension): ")

lines = open(f'./datasets/{lines_file}', encoding = 'utf-8', errors = 'ignore').read().split('\n')
convos = open(f'./datasets/{convos_file}', encoding = 'utf-8', errors = 'ignore').read().split('\n')


#max number of chars in propmt
QUESTION_LENGTH = int(input("enter max char count per line (longer lines ignored): "))
#max number of words in response


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


#parsing text files
exchange = []
for convo in convos:
    exchange.append(convo.split(' +++$+++ ')[-1][1:-1].replace("'",  "").replace(",","").split())

dialogues = {}

for line in lines:
    dialogues[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

#questions consist of strings of prompts
#answers consist of responses to prompts
#indices of questions and answers correspond (i.e. answers[0] is the response to questions[0])
questions = []
answers = []



for convo in exchange:
    for i in range(len(convo)- 1):
        questions.append(dialogues[convo[i]])
        answers.append(dialogues[convo[i+1]])

del(convo, convos, dialogues, exchange, i, line, lines)

processed_questions = []
processed_answers = []
for i in range(len(questions)):
    if len(questions[i]) < QUESTION_LENGTH:
        processed_questions.append(clean_text(questions[i]))
        processed_answers.append(clean_text(answers[i]))


for i in range(len(processed_answers)):
    processed_answers[i] = ' '.join(processed_answers[i].split()[:11])

del(questions, answers, i)

length = len(processed_answers)
MAX_LENGTH = int(input(f"enter size to trim data to (1 - {length}): "))

processed_answers = processed_answers[:MAX_LENGTH]
processed_questions = processed_questions[:MAX_LENGTH]

word_to_count = {}

for line in processed_questions:
    for word in line.split():
        if word not in word_to_count.keys():
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

for line in processed_answers:
    for word in line.split():
        if word not in word_to_count.keys():
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

del(word, line)

WORD_COUNT_THRESH = int(input("enter minimum word occurance for it to be added to training data (2 seems to work): "))

thresh = WORD_COUNT_THRESH
vocab = {}
word_num = 0

#only keeps words with occurences above thresh
for word, count in word_to_count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1

del(word_to_count, word, count, thresh)
del(word_num)



for i in range(len(processed_answers)):
    processed_answers[i] = '<SOS> ' + processed_answers[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

zero_key = get_key(0, vocab)

vocab[zero_key] = vocab['<PAD>']
vocab['<PAD>'] = 0

del(token, tokens)
del(x)


inv_vocab = {w:v for v, w in vocab.items()}

del(i)

encoder_input = []
for line in processed_questions:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    encoder_input.append(lst)

decoder_input = []
for line in processed_answers:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    decoder_input.append(lst)

del(processed_answers, processed_questions, line, lst, word)

encoder_input = pad_sequences(encoder_input, 13, padding = 'post', truncating = 'post')
decoder_input = pad_sequences(decoder_input, 13, padding = 'post', truncating = 'post')

decoder_final_output = []
for i in decoder_input:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, 13, padding = 'post', truncating= 'post')
#decoder_final_output = to_categorical(decoder_final_output, len(vocab))


payload = {"decoder_final_output": decoder_final_output, "vocab": vocab, "encoder_input": encoder_input, "decoder_input": decoder_input, "inv_vocab": inv_vocab}

name = input("name of data file: ")
with open(f"./processed_data/{name}.pickle","wb") as f:
    pickle.dump(payload, f)

