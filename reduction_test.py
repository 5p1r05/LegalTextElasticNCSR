import nltk
from nltk.tokenize import sent_tokenize
import os
import random

with open("clean_corpus/08_1.txt") as f:
    text = f.read()

sentences = sent_tokenize(text)

for sentence in sentences:
    print(sentence)
    print("################")
    print("------------")