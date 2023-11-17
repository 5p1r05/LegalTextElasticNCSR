import pandas as pd
from elasticsearch import Elasticsearch
import argparse
import json
from src.word2vec_embeddings import Word2Vec_Embeddings
from src.doc2vec_embeddings import Doc2Vec_Embeddings
from src.USE_embeddings import USE_Embeddings
from src.SBERT_embeddings import SBERT_Embeddings

import time
import sys

# Get info about the retrieved documents from the response object
def parse_response(response):
	similar_docs = []
	for hit in response.body['hits']['hits']:
		similar_doc = {
			"docID": hit['_source']['id'],
			"text": hit['_source']['text'],
			"similarity": hit['_score']
		}
		similar_docs.append(similar_doc)
	return similar_docs


# Parse the arguments
parser = argparse.ArgumentParser(
                    prog='retrieveDocuments.py',
                    description='make queries to the server')

parser.add_argument('-e', '--embeddings', help='provide the type of embeddings you want to use. The Valid types are : word2vec, doc2vec, USE, SBERT', nargs='+', default=['word2vec', 'doc2vec', 'USE', 'SBERT'])
parser.add_argument('-q', '--query_path', help='provide the path to the file that contains the text to be queried', required=True)
parser.add_argument('-s', '--save_path', help='provide the path to the file you want to save the json file that contains the results', required=True)
args = parser.parse_args()

# Check if the embedding names are valid
for embedding_type in args.embeddings:
	if(embedding_type not in ['word2vec', 'doc2vec', 'USE', 'SBERT']):
		print("ERROR: embedding types are not valid. Check -h for help")
		exit(1)

# Establish connection with the elasticsearch server
es = Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme':'http'}])

# Read the text in the query file
with open(args.query_path, "r") as f:
    query_text = f.read()

results = {}

# First retrieve the documents based on the default TF/IDF metric:
body = {
  "query": {
    "match": {
      "text": {
        "query": query_text
      }
    }
  }
}
response = es.search(index='legal_cases', body=body)

results['tf_idf'] = parse_response(response)


# Iterate over the embedding types and make a search request for each type
for embedding_type in args.embeddings:
	if(embedding_type == 'doc2vec'):
		vectorizer = Doc2Vec_Embeddings()
	elif(embedding_type == 'SBERT'):
		vectorizer = SBERT_Embeddings()
	elif(embedding_type == 'word2vec'):
		vectorizer = Word2Vec_Embeddings()
	elif(embedding_type == 'USE'):
		vectorizer = USE_Embeddings()
	
	# Get the embedding of the query text
	query_embedding = vectorizer.get_embedding(query_text)
  
	# Create the body of the search request
	body = {
	"query": {
		"script_score": {
		"query": {
			"match_all": {}
		},
		"script": {
			# Take the cosine similarity between the query vector and the vectors of the indexed documents
			"source": f"cosineSimilarity(params.queryVector, '{embedding_type}')+1.0",
			"params": {
			"queryVector": eval(query_embedding)
					}
				}
			}
		}
	}
	# Perform the request
	response = es.search(index='legal_cases', body=body)

	similar_docs = parse_response(response)

	# Add the info about the 10 most similar documents into the dictonary entry for each embedding type
	results[str(embedding_type)] = similar_docs


# Dump the dictionary into a json file
with open(args.save_path, "w") as f:
	json.dump(results, f)
