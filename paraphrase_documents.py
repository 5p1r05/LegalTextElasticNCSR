import argparse
import os

from src.change_synonyms_wordnet import ChangeSynonymsFromDict
from src.swap_letters import LetterSwapper
from src.text_reduction import TextReducer

parser = argparse.ArgumentParser(
                    prog='paraphraseDocuments',
                    description='different ways to change the contents of a text')

parser.add_argument('-s', '--source_path', help='provide the path to the folder containing the raw text files to be changed', required=True)
parser.add_argument('-d', '--destination_path', help='prodive the path to the destination folder that the changed text files will be stored', required=True)
parser.add_argument('-f', '--files_num', help='number of files to be changed. Can be a number or "all". The files are chosen at random', required=True)
parser.add_argument('-m', '--mode', help='the valid modes are: change_synonyms, swap_letters, reduce_text', required=True)
parser.add_argument('-p', '--change_percentage', help='percentage of the text to be changed', required=True)

args = parser.parse_args()


# Initialize the right paraphraser object
if(args.mode=='change_synonyms'):
    paraphraser = ChangeSynonymsFromDict(change_percentage=float(args.change_percentage))
elif(args.mode=='swap_letters'):
    paraphraser = LetterSwapper(swap_percentage=float(args.change_percentage))
elif(args.mode=='reduce_text'):
    paraphraser = TextReducer(reduce_percentage=float(args.change_percentage))
else:
    print("ERROR: mode is not valid. check -h for help")
    exit(1)


# Call the right paraphraser and let it change the files from the specified path
paraphraser.change_files(args.source_path, args.destination_path, files_to_change=args.files_num)

