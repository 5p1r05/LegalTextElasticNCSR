import requests
import nltk
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import requests
import time
import os

nltk.download('stopwords')

class ChangeSynonymsFromDict:
    
    def __init__(self, change_percentage=0.2):
        self.change_percentage = change_percentage
    
    def change_files(self, source_path, destination_path, files_to_change="all"):
        source_files = os.listdir(source_path)

        # Take the names of the files to be changed
        # If a specific number of files to be changed is specified
        # choose files randomly
        if(files_to_change != "all"):
            if(int(files_to_change) > len(source_files)):
                return -1
            source_files = random.sample(source_files, int(files_to_change))

        # Iterate over files and change words with their synonyms
        for file_name in source_files:
            with open(source_path + "/" + file_name, "r") as f:
                text = f.read()

            text = word_tokenize(text)

            # Get the indices of the words that are not stopwords and are english words.
            indices = []
            for i, word in enumerate(text):
                if(word not in stopwords.words('english')): # and word not in stopwords.words('english')
                    indices.append(i)

            # Select indices to be changed randomly based on the percentage
            indices = random.sample(indices, int(self.change_percentage*len(indices)))

            # Iterate over the sampled indices and change the corresponding words with their randomly chosen synonyms
            for i in indices:
                # r = requests.get(url=f'https://api.api-ninjas.com/v1/thesaurus?word={text[i]}')
                # synonyms = r.json()['synonyms']
                syns = wordnet.synsets(text[i])
                if(len(syns) == 0):
                    continue
                syn = random.choice(syns)
                synonym_word = random.choice(syn.lemmas()).name()
                text[i] = synonym_word

            finalized_text = " ".join(text)

            # Save the result text with the right name
            root_file_name = file_name.split(".")[0]
            with open(f"{destination_path}/{root_file_name}_simple_synonyms_{self.change_percentage}.txt", "w") as f:
                f.write(finalized_text)



