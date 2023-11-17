for mode in change_synonyms reduce_text swap_letters
do
	for percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
	do
		python3 ../paraphrase_documents.py -s ../clean_corpus/ -d paraphrased_corpus/$mode/$percentage/ -f "all" -m $mode -p $percentage
	done
done
