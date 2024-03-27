import os
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import spacy
import nltk
import re   # regular expression
import string
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

import scipy.sparse as sp
import joblib
import emoji
from autocorrect import Speller   # for correcting spelling
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize    # for tokenizing string into words
from nltk.stem import WordNetLemmatizer    # for lemmatizing words
from nltk.tag import pos_tag # for tagging words with their parts of speech (POS)