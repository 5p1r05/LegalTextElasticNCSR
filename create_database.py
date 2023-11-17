import pandas as pd
from elasticsearch import Elasticsearch
import argparse

parser = argparse.ArgumentParser(
                    prog='createDatabase',
                    description='create the elasticsearch database and add the documents along with their embeddings')

parser.add_argument('-d', '--data_path', help='enter the path to the .csv file that contains the embedding vectors', required = True)
parser.add_argument('-p', '--port', default=9200, help='enter the port which the elasticsearch server listens to')

# Parse the arguments
args = parser.parse_args()

# Establish connection with the elasticsearch server
es = Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme':'http'}])

# Delete the Database if it exists
es.options(ignore_status=[400,404]).indices.delete(index='legal_cases')

# Specify the mapping, which will describe the structure of each document
mappings = {
	"properties":{
		"id":{
				"type": "text"
		},
		"text":{
				"type":"text"
		},
		"doc2vec":{
				"type": "dense_vector",
				"dims": 300
		},
		"word2vec":{
				"type": "dense_vector",
				"dims": 300
		},
		"USE":{
				"type": "dense_vector",
				"dims": 512
		},
		"SBERT":{
				"type": "dense_vector",
				"dims": 768
		}    
    }
}

# Create the index
result = es.indices.create(index="legal_cases", mappings=mappings)

# Access the data that are going to be uploaded on the server
embeddings_df = pd.read_csv(str(args.data_path))

# Iterate over the rows of the csv and add each document on the database
for index, row in embeddings_df.iterrows():
	doc = { 
		'id' : str(row['ids']),
		'text' : str(row['raw_texts']),
		"doc2vec": eval(row['doc2vec_emb']),
		"word2vec": eval(row['word2vec_emb']),
		"USE": eval(row['USE_emb']),
		"SBERT": eval(row['SBERT_emb'])
	}

	res = es.index(index="legal_cases", id=index, document=doc)