import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    UniversalSentenceEncoder
)
import numpy as np
import os
import pandas as pd
import time

# A class that contains the spark-nlp pipeline for creating USE embeddings and helpful functions to produce them
class USE_Embeddings:

	def __init__(self):
		# Start Spark Session
		self.spark = sparknlp.start()

		# Step 1: Transforms raw texts to `document` annotation
		self.documentAssembler = DocumentAssembler() \
			.setInputCol('text') \
			.setOutputCol('document')

		self.embeddings = UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence_embeddings")

		# Define the pipeline
		self.pipeline = Pipeline() \
			.setStages([
			self.documentAssembler,
			self.embeddings])
	
	def get_embedding(self, text):
		start = time.time()

		# Transform the text into a spark dataframe
		data = self.spark.createDataFrame([[text]]).toDF("text")

		# Pass it through the pipeline
		model = self.pipeline.fit(data)

		# Pass it through the doc2vec model
		result = model.transform(data)
		end = time.time()
		print(end-start)
		return str(result.select("sentence_embeddings").collect()[0]['sentence_embeddings'][0].asDict()['embeddings'])

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
			
			embedding = self.get_embedding(text)

			# Unpack the output of the model inference and get the embedding
			embeddings.append(embedding)
			ids.append(file_name.split('.')[0])


		# Create a DataFrame that contains the ids and the corresponding embeddings
		USE_df = pd.DataFrame({
			"ids" : ids,
			"USE_emb": embeddings
		})
		return USE_df
		# Convert it into a csv
		#word2vec_df.to_csv("USE_embeddings.csv")
