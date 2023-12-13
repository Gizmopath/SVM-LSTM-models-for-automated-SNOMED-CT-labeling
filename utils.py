import re
import json
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from unidecode import unidecode

context_specific_stopwords = [
    "interessati",
    "interessante",
    "interesse",
    "score",
    "isup",
    "percentuale",
    "gruppo",
    "frammenti",
    "interessati",
    "estensione",
    "lineare",
    "evidente",
    "tale",
    "deve",
    "essere",
    "tratti",
    "prima",
    "ipotesi",
    "dati",
    "laboratorio",
    "diagnostica",
    "clinici",
    "confermare",
    "tuttavia",
    "esame",
    "atto",
    "utile",
    "eventuale",
    "referto",
    "seguir",
    "valutazione",
    "immagini",
    "meno",
    "verosimile",
    "sospetto",
    "appare",
    "caratteristiche",
    "definizione",
    "riconducibili",
    "controllo",
    "talora",
    "senza",
    "marcato",
    "osservano",
    "destra",
    "sinistra",
    "adeguata",
    "suggestive",
    "quali",
]

stemmer = SnowballStemmer('italian')

def remove_digits(string):
    return re.sub(r'\d', ' ', string)

def replace_concatenated_words(string):
    return re.sub(r'([a-z]+)([A-Z][a-z]+)', r'\1 \2', string).capitalize()

def clean(text):
    text = unidecode(text)
    text = remove_digits(text)
    text = text.replace("/", "").replace("-", "")
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = replace_concatenated_words(text)
    text = text.lower()
    text = unidecode(text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('italian') + stopwords.words('english') + context_specific_stopwords)
    filtered_tokens = [token for token in tokens if (token not in stop_words and not re.search("\d", token) and len(token) > 3)]
    lemmas = [stemmer.stem(token) for token in filtered_tokens]

    return " ".join(lemmas)

def save_vocabulary(vocabulary):
    print(f"Dimensione del vocabolario: {str(len(vocabulary))}")
    with open("vocabolario.json", "w") as f:
        f.write(json.dumps(vocabulary))
