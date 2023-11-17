import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    Tokenizer,
    WordEmbeddingsModel,
    Word2VecModel
)
import numpy as np
import os
import pandas as pd
import time

# A class that contains the spark-nlp pipeline for creating Word2Vec embeddings and helpful functions to produce them
class Word2Vec_Embeddings:
	def __init__(self):

		# Start Spark Session
		self.spark = sparknlp.start()

		# Transform raw texts to `document` annotation. (more info at the spark nlp documentation)
		self.documentAssembler = DocumentAssembler() \
			.setInputCol('text') \
			.setOutputCol('document')

		# Tokenization
		self.tokenizer = Tokenizer() \
			.setInputCols(['document']) \
			.setOutputCol('token')

		# Generate the Embeddings
		self.embeddings = Word2VecModel.pretrained() \
			.setInputCols(["token"]) \
			.setOutputCol("embeddings")

		# Define the pipeline
		self.pipeline = Pipeline() \
			.setStages([
			self.documentAssembler,
			self.tokenizer,
			self.embeddings])

	# Pass a single document through the pipeline
	def get_embedding(self, text):
		start = time.time()
		# Transform the text into a spark dataframe
		data = self.spark.createDataFrame([[text]]).toDF("text")

		# Pass it through the pipeline
		model = self.pipeline.fit(data)

		
		# Pass it through the word2vec model
		result = model.transform(data)

		# Word2Vec produces word-level embeddings.
		# So, we produce a vector which is the average of all the word vectors.
		

		word_embeddings = []
		for embeddings_dict in result.select('embeddings').collect()[0]['embeddings']:
			word_embedding = embeddings_dict.asDict()['embeddings']

			# Add the word embedding to the list only if it is not an OOV word
			if(word_embedding[0] != 0.0):
				word_embeddings.append(word_embedding)

		document_embedding = np.mean(word_embeddings, axis=0)
		end = time.time()
		print("WORD2VEC EMBEDDING TIME")
		print(end-start)
		return str(document_embedding.tolist())
	

	# Pass multiple documents through the pipeline. (Returns a DataFrame)
	def get_embeddings_bulk(self, path):
		# List all the document names
		file_names = os.listdir(path)

		ids = []
		embeddings = []

		# Iterate over files and get their text
		for i, file_name in enumerate(file_names):
			with open(path + file_name, "r") as f:
				text = f.read()
		
			document_embedding = self.get_embedding(text)
			# Unpack the output of the model inference and get the embedding
			embeddings.append(document_embedding)
			ids.append(file_name.split('.')[0])


		# Create a DataFrame that contains the ids and the corresponding embeddings
		word2vec_df = pd.DataFrame({
			"ids" : ids,
			"word2vec_emb": embeddings
		})
		
		return word2vec_df
		

