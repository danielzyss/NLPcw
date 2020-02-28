import os
import subprocess
import sys
import io
import numpy as np
import torch
import re, collections
from collections import Counter, defaultdict
import random
import tqdm
import unicodedata
import time
from bpemb import BPEmb
from sklearn.manifold import TSNE
import pandas

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

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr as pearson
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from zipfile import ZipFile
from sklearn.metrics import mean_absolute_error

