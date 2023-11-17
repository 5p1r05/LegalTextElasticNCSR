import os
import random

sampled_files_num = 10

paraphrase_types = ["change_synonyms", "reduce_text", "swap_letters"]
paraphrase_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


for test_num in range(1, 10):
    if( not os.path.isdir(f'../tests/results{test_num}')):
        os.mkdir(f"../tests/results{test_num}")
    
    for paraphrase_type in paraphrase_types:
        if (not os.path.isdir(f'../tests/results{test_num}/{paraphrase_type}')):
                os.mkdir(f"../tests/results{test_num}/{paraphrase_type}")

        for paraphrase_percentage in paraphrase_percentages:
            if (not os.path.isdir(f'../tests/results{test_num}/{paraphrase_type}/{paraphrase_percentage}')):
                os.mkdir(f"../tests/results{test_num}/{paraphrase_type}/{paraphrase_percentage}")

            query_docs_names = os.listdir(f"paraphrased_corpus/{paraphrase_type}/{paraphrase_percentage}/")
            sampled_names = random.sample(query_docs_names, sampled_files_num)

            for sampled_name in sampled_names:
                result_name = sampled_name.split(".txt")[0] + "_result.json"
                if(len(os.listdir(f"../tests/results{test_num}/{paraphrase_type}/{paraphrase_percentage}/")) < sampled_files_num):
                    print("destination")
                    print(f"../tests/results{test_num}/{paraphrase_type}/{paraphrase_percentage}/{result_name}")
                    os.system(f"python3 ../retrieve_documents.py -q paraphrased_corpus/{paraphrase_type}/{paraphrase_percentage}/{sampled_name} -s ../tests/results{test_num}/{paraphrase_type}/{paraphrase_percentage}/{result_name}")

