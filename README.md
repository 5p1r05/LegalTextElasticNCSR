# LegalTextElasticSearch

The abundance of Information nowadays is a blessing but also a curse. Lawyers, like most professions, utilize data to make better decisions. More specifically, they analyze past court records to get the information they need for their future cases.

One obvious problem that they need to tackle is how to navigate the vast pool of data and land on the information that they are looking for.
Recent advancements in the field of AI and deep learning could help with that.

In this project, I implemented a system that stores legal documents in vector forms by leveraging various embedding techniques and it lets the user make queries and retrieve the most relevant document. I made use of the elastic search engine that promises real-time search, scalability, reliability, and flexibility.

This README contains instructions about producing embeddings, seting up the server, filling it with documents, modifying the documents and retrieving them.
In the end, it explains how to evaluate this system.

## Set up the elasticsearch server

**Step 1:** Download docker desktop by following the instructions [here](https://docs.docker.com/desktop/install/linux-install/)

**Step 2:** Go to settings->resources and increase the memory to 4GB (minimum)

**Step 3:** Run the following command to create the elastic network:

```
docker network create elastic
```

**Step 4:** Run the following command to download and run the docker image

```
docker run --name elasticsearch -m 4G --net elastic -p 9200:9200 -e discovery.type=single-node -e ES_JAVA_OPTS="-Xms4g -Xmx4g" -e xpack.security.enabled=false -v {LOCAL_FOLDER}:/usr/share/elasticsearch/data -it    docker.elastic.co/elasticsearch/elasticsearch:8.8.2
```

**LOCAL_FOLDER:** replace this with a folder in your computer that will bind to the docker container and will store the files locally

The Server should be up and Running an it should be listening to the port 9200.

To check., visit [http://localhost:9200/](http://localhost:9200/) on your browser.

You should be able to see a message indicating the version of elasticsearch along with additional information.

In order to start and stop the elasticsearch server each time, open the docker desktop app and controll it from there.

### *Initialize the environment*

Type the following command into the terminal:

```bash
conda env create -f env.yaml -n {environment_name}
```

Now that the environment is created, activate it by typing

```
conda activate {environment_name}
```

### Create the embeddings

(Download the documents from here [https://drive.google.com/drive/folders/1i8jiwQqXZUUsyRTKIK_4YBrpozWqv4OZ?usp=sharing]())

Create embeddings for text files that are into a single folder

usage:

```
python3 createEmbeddings.py  [-e EMBEDDINGS [EMBEDDINGS ...]] -c CORPUS_PATH -s SAVE_PATH
```

**EMBEDDINGS:** The types of embeddings you want to produce. Valid embedding types are:

    word2vec, doc2vec, USE , SBERT (you can provide more than one. The default parameter is all for of them)

**CORPUS_PATH:** The path of the folder that contains the raw text files.

**SAVE_PATH:** The path of the csv file that will store the embeddings of the files along with additional information.

example run:

```
python3 create_embeddings.py --embeddings doc2vec word2vec USE SBERT --corpus_path clean_corpus/ --save_path embeddings/combined_embeddings.csv
```

### Create the Database and add the files

Connect to an elasticsearch server and index the documents along with their embeddings

usage:

```
python3 createDatabase.py -d DATA_PATH [-p PORT]
```

**DATA_PATH:** The path to the csv file that contains the embeddings. (The ones created by createEmbeddings.py)

**PORT:** The port  on which the server is binded (The default is port 9200)

example run

```
python3 create_database.py -d embeddings/combined_embeddings.csv
```

### Make a text query

produce different embeddings for a query document and make a vector search on the server. Using the cosine similarity as the metric, retrieve the most relevant documents.

usage:

```
python3 retrieve_documents.py [-h] [-e EMBEDDINGS [EMBEDDINGS ...]] -q QUERY_PATH -s SAVE_PATH
```

**EMBEDDINGS:** The types of embeddings you want to produce for the query text. Valid embedding types are:

    word2vec, doc2vec, USE , SBERT (you can provide more than one. The default parameter is all for of them)

**QUERY_PATH:** The path to the file that contains the text you want to use for the query

**SAVE_PATH:** The path to the json file that the results will be saved.

example run

```
python3 retrieve_documents.py -q clean_corpus/08_1.txt -s results/QueryResults.json
```

## Paraphrase Documents

Make various changes to the documents. This can be useful for the evaluation of the document retrieval system

usage:

```
python3 paraphraseDocuments.py [-h] -s SOURCE_PATH -d DESTINATION_PATH -f FILES_NUM -m MODE -p
                           CHANGE_PERCENTAGE

different ways to change the contents of a text

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE_PATH, --source_path SOURCE_PATH
                        provide the path to the folder containing the raw text files to be changed
  -d DESTINATION_PATH, --destination_path DESTINATION_PATH
                        prodive the path to the destination folder that the changed text files will be
                        stored
  -f FILES_NUM, --files_num FILES_NUM
                        number of files to be changed. Can be a number or "all". The files are chosen
                        at random
  -m MODE, --mode MODE  the valid modes are: change_synonyms, swap_letters, reduce_text
  -p CHANGE_PERCENTAGE, --change_percentage CHANGE_PERCENTAGE
                        percentage of the text to be changed

```

**MODES:**

* *change_synonyms*: Select random words from the text, based on the change percentage and substitute them with random synonyms from the dict.
* *swap_letters*: Select random words based on the change percentage and swap two letters randomly
* *reduce_text: Retain a subset of the text and discard the rest, also based on the change percentage*

# Evaluation

**step 1**: Go to the evaluation folder and run the paraphrase_documents_script script. This modifies all the documents in the ../clean_corpus/ directory using all possible ways and using different values for the hyperparameter.
 It stores them in the paraphrased_corpus/ directory

**step 2**: Run the bulk_queries.py program. This program samples documents from the paraphrased_corpus/ directory,
performs queries using them and stores the results into the ../tests/ folder.

**step 3**: Run the evaluata.py program that accesses the results produced above and generates plots that measure the performance of the system using the mean reciprocal rank metric.
