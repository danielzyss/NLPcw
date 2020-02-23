import os
import subprocess
import sys
import io
import numpy as np
import torch
import re, collections
from collections import Counter, defaultdict
import random

import spacy
from nltk import download
import nltk
try:
    nltk.data.find('tokenizers/stopwords')
except:
    download('stopwords') #stopwords dictionary, run once
from nltk.corpus import stopwords

stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))
try:
    nlp_de = spacy.load('de300')
except:
    os.system(sys.executable + " -m spacy download de_core_news_sm")
    os.system(sys.executable + " -m spacy link de_core_news_sm de300")
    nlp_de = spacy.load('de300')

try:
    nlp_en =spacy.load('en300')
except:
    os.system(sys.executable + " -m spacy download en_core_web_sm")
    os.system(sys.executable + " -m spacy link en_core_web_sm en300")
    nlp_en = spacy.load('en300')


