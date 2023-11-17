import argparse
import numpy as np
import os
import pandas as pd
from functools import reduce

from src.word2vec_embeddings import Word2Vec_Embeddings
from src.doc2vec_embeddings import Doc2Vec_Embeddings
from src.USE_embeddings import USE_Embeddings
from src.SBERT_embeddings import SBERT_Embeddings

parser = argparse.ArgumentParser(
                    prog='createEmbeddings',
                    description='create embeddings for the provided text',
                    epilog='Text at the bottom of help')

parser.add_argument('-e', '--embeddings', help='provide the type of embeddings you want to produce. The Valid types are : word2vec, doc2vec, USE, SBERT', nargs='+', default=['word2vec', 'doc2vec', 'USE', 'SBERT'])
parser.add_argument('-c', '--corpus_path', help='provide the path to the folder containing the raw text files', required=True)
parser.add_argument('-s', '--save_path', help='provide the path to the folder you want to save the embeddings', required=True)

# Parse the arguments
args = parser.parse_args()
embeddings_list = args.embeddings

# Check if the embedding names are valid
for embedding_type in embeddings_list:
    if(embedding_type not in ['word2vec', 'doc2vec', 'USE', 'SBERT']):
        print("ERROR: embedding types are not valid. Check -h for help")
        exit(1)

# List all the document names
file_names = os.listdir(args.corpus_path)

raw_texts = []
ids = []

for i, file_name in enumerate(file_names):
	with open(args.corpus_path + file_name, "r") as f:
		text = f.read()

	raw_texts.append(text)
	ids.append(file_name.split('.')[0])


# Create a dataframe only with the raw text and the ids
embeddings_df = pd.DataFrame({
	"ids": ids,
	"raw_texts": raw_texts
})


vectorizers = []
# Iterate over the embedding types and initialize the vectorizer objects
for embedding_type in embeddings_list:
	if(embedding_type == 'doc2vec'):
		vectorizers.append(Doc2Vec_Embeddings())
	elif(embedding_type == 'SBERT'):
		vectorizers.append(SBERT_Embeddings())
	elif(embedding_type == 'word2vec'):
		vectorizers.append(Word2Vec_Embeddings())
	elif(embedding_type == 'USE'):
		vectorizers.append(USE_Embeddings())

for vectorizer in vectorizers:
	new_dataframe = vectorizer.get_embeddings_bulk(args.corpus_path)

	# Call the function that corresponds to each embedding type
	# function = 'create_' + embedding_type + '_embeddings(\'' + args.corpus_path  + '\')'   
	# new_dataframe = eval(function)
	embeddings_df = pd.merge(embeddings_df, new_dataframe, on="ids")


embeddings_df.to_csv(args.save_path, index=False)
# print(files_path)
# print(embeddings_list)

