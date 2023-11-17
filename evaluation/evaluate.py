import json
import os
import matplotlib.pyplot as plt
import numpy as np 

paraphrase_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
embedding_types = ["tf_idf", "word2vec", "doc2vec", "USE", "SBERT"]

def get_doc_id(file_name):
    return "_".join(file_name.split("_")[0:2])


def evaluate_generate_plots(folder = "result", paraphrase_types = ["change_synonyms", "reduce_text", "swap_letters"]):

    # paraphrase_types = ["change_synonyms", "reduce_text", "swap_letters"]
    # paraphrase_types = ['reduce_text']
    

    score_info = {}

    for paraphrase_type in paraphrase_types:

        score_info[str(paraphrase_type)] = {}

        for paraphrase_percentage in paraphrase_percentages:
            current_path = f"../tests/{folder}/{paraphrase_type}/{paraphrase_percentage}/"
            file_names = os.listdir(current_path)
            
            score = 0
            score_info[str(paraphrase_type)][str(paraphrase_percentage)] = {}


            for file_name in file_names:
                with open(current_path + file_name, "r") as f:
                    results_json = json.loads(f.read())

                mrr_results = {}
                query_id = get_doc_id(file_name)
                
                for embedding_type in results_json.keys():
                    if(str(embedding_type) not in score_info[str(paraphrase_type)][str(paraphrase_percentage)].keys()):
                        score_info[str(paraphrase_type)][str(paraphrase_percentage)][str(embedding_type)] = []     
                    for i, retrieved_doc in enumerate(results_json[embedding_type]):
                        if(retrieved_doc["docID"] == query_id):
                            score = 1/(i+1)
                            score_info[str(paraphrase_type)][str(paraphrase_percentage)][str(embedding_type)].append(score)
                    



    embeddings_scores = {}

    for embedding_type in embedding_types:
        embeddings_scores[str(embedding_type)] = {}

    for paraphrase_type in paraphrase_types:
        for paraphrase_percentage in paraphrase_percentages:
            for embedding_type in embedding_types:
                if(str(paraphrase_percentage) not in embeddings_scores[str(embedding_type)].keys()):
                    embeddings_scores[str(embedding_type)][str(paraphrase_percentage)] = 0
                embeddings_scores[str(embedding_type)][str(paraphrase_percentage)] += sum(score_info[str(paraphrase_type)][str(paraphrase_percentage)][str(embedding_type)])/30



    return embeddings_scores



   

paraphrase_types = ["change_synonyms", "reduce_text", "swap_letters"]
#paraphrase_types = ["reduce_text"]
results_num = 5

results = {}
for paraphrase_type in paraphrase_types:
    print(f"{paraphrase_type}: ")
    for result_num in range(1, results_num + 1):
        results[f"results{result_num}"] = evaluate_generate_plots(f"results{result_num}", paraphrase_types=[paraphrase_type])
    
    # Now take the mean
    average_scores = {}

    for embedding_type in embedding_types:

        average_scores[embedding_type] = []
        stds = []

        for paraphrase_percentage in paraphrase_percentages:
            average_list = []

            # Iterate over the tests
            for test in results.keys():
                average_list.append(float(results[test][embedding_type][str(paraphrase_percentage)]))

            # Add the average to the list
            average_scores[embedding_type].append(np.mean(average_list))
            stds.append(np.std(average_list))
        
        print(f"    {embedding_type}:")
        std = np.mean(stds)
        print(f"        {std}")
        

        plt.plot(paraphrase_percentages, average_scores[embedding_type], label=str(embedding_type) if not embedding_type == "tf_idf" else "bm25")
        plt.scatter(paraphrase_percentages, average_scores[embedding_type])

    plt.title(str(paraphrase_type))
    plt.xlabel("percentage of change")
    plt.ylabel("MRR")

    plt.legend()
    plt.show()
                
        
    with open("resultsfile.json", "w") as f:
        json.dump(results, f)

