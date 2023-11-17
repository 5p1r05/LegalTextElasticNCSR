import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    BertSentenceEmbeddings
)
import pandas as pd
import os
import time

# A class that contains the spark-nlp pipeline for creating SBERT embeddings and helpful functions to produce them
class SBERT_Embeddings:
	def __init__(self):
		
		# Start Spark Session
		self.spark = sparknlp.start()

		# Transform raw texts to `document` annotation. (more info at the spark nlp documentation)
		self.documentAssembler = DocumentAssembler() \
			.setInputCol("text") \
			.setOutputCol("document")

		# Load the pre-trained BERT model for sentence embeddings
		self.embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased_legal", "en") \
			.setInputCols(["document"]) \
			.setOutputCol("sbert_embeddings")


		# Define the pipeline
		self.pipeline = Pipeline() \
			.setStages([
				self.documentAssembler,
				self.embeddings])

	# Pass a single document through the pipeline
	def get_embedding(self, text):
		start = time.time()
		# Transform the text into a spark dataframe
		data = self.spark.createDataFrame([[text]]).toDF("text")

		# Pass it through the pipeline
		model = self.pipeline.fit(data)

		# Pass it through the doc2vec model
		result = model.transform(data).toPandas()
		end = time.time()

		print("SBERT")
		print(end-start)

		# Unpack the output of the model inference, get the embedding and return it
		return str(result["sbert_embeddings"][0][0]["embeddings"])

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
		sbert_df = pd.DataFrame({
			"ids" : ids,
			"SBERT_emb": embeddings
		})

		return sbert_df
