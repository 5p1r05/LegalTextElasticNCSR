import nltk
import os
import random
from nltk.tokenize import word_tokenize

def swap_letters(word, pos1, pos2):
    word[pos1], word[pos2] = word[pos2], word[pos1]
    return word


class LetterSwapper:
    
    def __init__(self, swap_percentage=0.2):
        self.swap_percentage=swap_percentage
    
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

            # Get the valid indices of tokens
            indices = range(0, len(tokens))

            # Swap indices randomly
            sampled_indices = random.sample(indices, int(self.swap_percentage*len(indices)))


            for i in sampled_indices:
                token = tokens[i]
                if(len(token) <= 1):
                    continue
                letter_indices = list(range(0, len(token)))
                pos1 = random.choice(letter_indices)
                letter_indices.remove(pos1)
                pos2 = random.choice(letter_indices)

                swapped_token = swap_letters(list(token), pos1, pos2)
                tokens[i] = "".join(swapped_token)

            finalized_text = " ".join(tokens)

            root_file_name = file_name.split(".")[0]
            with open(f"{destination_path}/{root_file_name}_swap_letters_{self.swap_percentage}.txt", "w") as f:
                f.write(finalized_text)
            


# paraphraser = LetterSwapper(swap_percentage=1)

# paraphraser.change_files("clean_corpus/", "paraphrased_corpus", files_to_change=1)

                



        