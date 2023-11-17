from sparknlp.base import DocumentAssembler, Pipeline, LightPipeline, EmbeddingsFinisher
import sparknlp
import os
import pandas as pd
from sparknlp.annotator import (
    Tokenizer,
    Normalizer,
    StopWordsCleaner,
    Doc2VecModel
)
import time

# A class that contains the spark-nlp pipeline for creating Doc2Vec embeddings and helpful function to produce them
class Doc2Vec_Embeddings:

	def __init__(self):

		#start a sparknlp session
		self.spark = sparknlp.start()

		# Create modules that preprocess the text
		self.document = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

		# Tokenization
		self.token = Tokenizer()\
		.setInputCols("document")\
		.setOutputCol("token")

		# Normalization
		self.norm = Normalizer()\
		.setInputCols(["token"])\
		.setOutputCol("normalized")\
		.setLowercase(True)

		# Remove stop words
		self.stops = StopWordsCleaner.pretrained()\
		.setInputCols("normalized")\
		.setOutputCol("cleanedToken")

		# Generate embeddings
		self.doc2Vec = Doc2VecModel.pretrained("doc2vec_gigaword_wiki_300", "en")\
			.setInputCols("cleanedToken")\
			.setOutputCol("sentence_embeddings")

		# Compine the models into a single pipeline
		self.pipeline = Pipeline() \
			.setStages([
			self.document,
			self.token,
			self.norm,
			self.stops,
			self.doc2Vec
			])

	# Pass a single document through the pipeline
	def get_embedding(self, text):
		start = time.time()
		# Transform the text into a spark dataframe
		data = self.spark.createDataFrame([[text]]).toDF("text")

		# Pass it through the pipeline
		model = self.pipeline.fit(data)

		# Pass it through the doc2vec model
		result = model.transform(data)
		end = time.time()
		print("DOC2vec embedding time")
		print(end-start)
		# Unpack the output of the model inference, get the embedding and return it
		return str(result.select('sentence_embeddings').collect()[0]['sentence_embeddings'][0]['embeddings'])

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

			embeddings.append(embedding)
			ids.append(file_name.split('.')[0])

		# Create a DataFrame that contains the ids and the corresponding embeddings
		doc2vec_df = pd.DataFrame({
			"ids" : ids,
			"doc2vec_emb": embeddings
		})
		return doc2vec_df