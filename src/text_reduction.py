import nltk
from nltk.tokenize import word_tokenize
import os
import random

class TextReducer:
    # Percentage of text that will be removed
    def __init__(self, reduce_percentage=0.8):
        self.reduce_percentage = reduce_percentage

    def change_files(self, source_path, destination_path, files_to_change="all"):
        source_files = os.listdir(source_path)

        # Take the names of the files to be changed
        # If a specific number of files to be changed is specified
        # choose files randomly
        if(files_to_change != "all"):
            if(int(files_to_change) > len(source_files)):
                return -1
            source_files = random.sample(source_files, int(files_to_change))
        
        for file_name in source_files:
            with open(source_path + "/" + file_name, "r") as f:
                text = f.read()

            tokens = word_tokenize(text)
            
            # Retain the tokens in the beginning of the text
            # And discart the rest
            token_num = int(1 - len(tokens)*self.reduce_percentage)
            tokens = tokens[:token_num]

            finalized_text = " ".join(tokens)

            # Save the result text with the right name
            root_file_name = file_name.split(".")[0]
            with open(f"{destination_path}/{root_file_name}_text_reduction_{self.reduce_percentage}.txt", "w") as f:
                f.write(finalized_text)

# paraphraser = TextReducer(reduce_percentage=0.01)
# paraphraser.change_files("clean_corpus", "paraphrased_corpus", files_to_change=1)

