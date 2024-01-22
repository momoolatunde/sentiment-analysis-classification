import pandas as pd
import spacy
from bs4 import BeautifulSoup
import re

# load SpaCy model
nlp = spacy.load('en_core_web_sm')

# preprocessing function
def preprocess_basic(text):
    # convert text to lowercase
    text = text.lower()

    # remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # handle negations (e.g., isn't -> is not)
    contractions_dict = {
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }
    contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    text = contractions_pattern.sub(lambda x: contractions_dict[x.group()], text)

    # expand acronyms
    acronyms_dict = {
        'afaik': 'as far as i know',  
        'afk': 'away from keyboard',
        'asap': 'as soon as possible',
        'atk': 'at the keyboard',
        'atm': 'at the moment',
        'a3': 'anytime, anywhere, anyplace',
        'bak': 'back at keyboard',
        'bbl': 'be back later',
        'bbs': 'be back soon',
        'bfn': 'bye for now',
        'b4n': 'bye for now',
        'brb': 'be right back',
        'brt': 'be right there',
        'btw': 'by the way',
        'b4': 'before',
        'b4n': 'bye for now',
        'cu': 'see you',
        'cul8r': 'see you later',
        'cya': 'see you',
        'faq': 'frequently asked questions',
        'fc': 'fingers crossed',
        'fwiw': 'for what it\'s worth',
        'fyi': 'for your information',
        'gal': 'get a life',
        'gg': 'good game',
        'gn': 'good night',
        'gmta': 'great minds think alike',
        'gr8': 'great!',
        'g9': 'genius',
        'ic': 'i see',
        'icq': 'i seek you (also a chat program)',
        'ilu': 'ilu: i love you',
        'imho': 'in my honest/humble opinion',
        'imo': 'in my opinion',
        'iow': 'in other words',
        'irl': 'in real life',
        'kiss': 'keep it simple, stupid',
        'ldr': 'long distance relationship',
        'lmao': 'laugh my a.. off',
        'lol': 'laughing out loud',
        'ltns': 'long time no see',
        'l8r': 'later',
        'mte': 'my thoughts exactly',
        'm8': 'mate',
        'nrn': 'no reply necessary',
        'oic': 'oh i see',
        'pita': 'pain in the a..',
        'prt': 'party',
        'prw': 'parents are watching',
        'qpsa?': 'que pasa?',
        'rofl': 'rolling on the floor laughing',
        'roflol': 'rolling on the floor laughing out loud',
        'rotflmao': 'rolling on the floor laughing my a.. off',
        'sk8': 'skate',
        'stats': 'your sex and age',
        'asl': 'age, sex, location',
        'thx': 'thank you',
        'ttfn': 'ta-ta for now!',
        'ttyl': 'talk to you later',
        'u': 'you',
        'u2': 'you too',
        'u4e': 'yours for ever',
        'wb': 'welcome back',
        'wtf': 'what the f...',
        'wtg': 'way to go!',
        'wuf': 'where are you from?',
        'w8': 'wait...',
        '7k': 'sick:-d laugher',
        'tfw': 'that feeling when',
        'mfw': 'my face when',
        'mrw': 'my reaction when',
        'ifyp': 'i feel your pain',
        'lol': 'laughing out loud',
        'tntl': 'trying not to laugh',
        'jk': 'just kidding',
        'idc': 'i don’t care',
        'ily': 'i love you',
        'imu': 'i miss you',
        'adih': 'another day in hell',
        'idc': 'i don’t care',
        'zzz': 'sleeping, bored, tired',
        'wywh': 'wish you were here',
        'time': 'tears in my eyes',
        'bae': 'before anyone else',
        'fimh': 'forever in my heart',
        'bsaaw': 'big smile and a wink',
        'bwl': 'bursting with laughter',
        'lmao': 'laughing my a** off',
        'bff': 'best friends forever',
        'csl': 'can’t stop laughing'
    }
    acronyms_pattern = re.compile(r'\b(' + '|'.join(acronyms_dict.keys()) + r')\b')
    text = acronyms_pattern.sub(lambda x: acronyms_dict[x.group()], text)

    # tokenize and remove punctuation
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct]

 # further cleaning (like removing non-alphabetic characters)
    cleaned_tokens = [token for token in tokens if token.isalpha()]

    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

