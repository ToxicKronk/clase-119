# Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# Función para añadir palabras raíz (stem words)
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Agregar todas las palabras de los patrones a una lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Agregar todas las etiquetas a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0]) 
print(classes)   

# Crear un corpus de palabras para el chatbot
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

training_data = []
number_of_tags = len(classes) 
labels = [0]*number_of_tags

# Crear una bolsa de palabras
for word_tags in word_tags_list:
     w_bag = []
     pattern_words = word_tags [0]
     for w_tags2 in pattern_words:
          index = pattern_words.index(w_tags2)
          word = stemmer.stem(w_tags2.lower())
          pattern_words[index] = word
     for w_tags3 in stem_words:
          if w_tags3 in pattern_words:
               w_bag.append(1)
          else:
               w_bag.append(0)
     print("bolsa de palabras", w_bag)
     labels_uncoding = list(labels)
     tag = word_tags[1]
     tag_index = classes.index(tag)
     labels_uncoding[tag_index] = 1
     training_data.append([w_bag, labels_uncoding])
print(training_data[0])     
# Crear datos de entrenamiento

def d_training (training_data):
     training_data = np.array(training_data, dtype=object)
     train_x = list(training_data[:,0])
     train_y = list(training_data[:,1])
     print("oraciones", train_x)
     print("etiquetas", train_y)
     return train_x,train_y

train_x, train_y = d_training(training_data)