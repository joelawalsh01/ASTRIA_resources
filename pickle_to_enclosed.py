import pickle
import pandas as pd
import sys
import glob
import os
from collections import Counter
import time
import torch
import argparse
from transformers import AutoTokenizer,AutoModelForTokenClassification
from transformers import pipeline
from nltk.tokenize import sent_tokenize


def _parse_args():
	"""
	Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
	are provided for convenience.
	:return: the parsed args bundle
	"""
	parser = argparse.ArgumentParser(description='trainer.py')
	parser.add_argument('--corpus_path', type=str, default='non', help='path to folder containing corpus')
	parser.add_argument('--dfpickled_path', type=str, default='non', help='path to pickled UCS dataframe')
	args = parser.parse_args()
	return args


def pd_to_triples(pickled_df_path):
	"""
	:param pickled: pickled pandas dataframe from the Union of Concerned Scientists datset
	:return: List[tuple] , tuple is (object, relation, country)
	"""
	
	df = pd.read_pickle(pickled_df_path)

	satellites = df['Current Official Name of Satellite']

	tuples = []
	for i,satellite in enumerate(satellites):
		un_registry_tuple = (satellite, 'unRregistry', df.loc[i]['Country/Org of UN Registry'])    
		launch_site_tuple = ( satellite, 'launchsiteCountry',df.loc[i]['Launch site Country'])


		operator = df.loc[i]['Country of Operator/Owner']


		if ('/') in operator:
			#print(str(i))
			#print("split")
			#print(operator)
			list_operator_countries = operator.split('/')
			#print(list_operator_countries)
			for country in list_operator_countries:
				operator_tuple = (satellite, 'operatorCountry', country)
				#print(operator_tuple)
				tuples.append(operator_tuple)

		if ('/') not in operator:
			operator_tuple = (satellite, 'operatorCountry',df.loc[i]['Country of Operator/Owner'])
			tuples.append(operator_tuple)

		contractor = df.loc[i]['Country of Contractor']
		#print(str(i))

		if ('/') in contractor:
			#print("split")
			#print(contractor)
			list_contractor_countries = contractor.split('/')
			#print(list_contractor_countries)
			for country in list_contractor_countries:
				contractor_tuple = (satellite, 'contractorCountry', country)
				#print(contractor_tuple)
				tuples.append(contractor_tuple)
		if ('/') not in contractor: 

			contractor_tuple = (satellite, 'contractorCountry', df.loc[i]['Country of Contractor'])                                           
			tuples.append(contractor_tuple)



		tuples.append(un_registry_tuple)

		tuples.append(launch_site_tuple)
		
	return tuples


def list_objects_generator(pickled_triples):
	triple_counter = Counter()
	for triple in triples:
		triple_counter.update({triple[0]})
	triples_list = list(triple_counter.keys())
	return triples_list


def enclosed_sentences(file_list, objects_list,nlp,num_intervals, target_dir):
	start = time.time()
	entities_counter = Counter()
	enclosed_sentences = []
	
	segment_length =len(file_list)// num_intervals

	for i,file in enumerate(file_list):
		

		my_text =open(file)
		full_string = my_text.read()

		# tokenize sentences, if entity is in sentence, keep sentence
		sentence_tokens = sent_tokenize(full_string)



		for sentence in sentence_tokens:

			entities = nlp(sentence)

			for entity in entities:
				if entity['word'] in objects_list:
					enclosed_sentences.append(sentence)


		end = time.time()
		print ("Time elapsed:", end - start)\
		
		if i%segment_length ==0:
			file_name = target_dir + "enclosed_sentences" + str(i)+ ".pkl"

			open_file = open(file_name, "wb")
			pickle.dump(enclosed_sentences, open_file)
			print("checkpoint")
			open_file.close()
			


	end = time.time()
	print ("Time elapsed:", end - start)
	return enclosed_sentences

def file_list_gen(path_name):

    os.chdir(path_name)
    file_list = []
    for file in glob.glob('*.txt'):
        file_list.append(file)
    return file_list





if __name__ == '__main__':
	start_time = time.time()
	args = _parse_args()


	# Initialize BERT model for tokenization and NER 

	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
	NER_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


	# aggregation strategy: if left undefined, will default to breaking up some entities into subwords
	if torch.cuda.is_available()== True:
		nlp = pipeline('ner',tokenizer = tokenizer,  model = NER_model, aggregation_strategy = "max", device = 0 )
	else:
		nlp = pipeline('ner',tokenizer = tokenizer,  model = NER_model, aggregation_strategy = "max")

	# read in pickle of cleaned df from UCS, extract  triples

	pickled_df_path = args.dfpickled_path
	triples = pd_to_triples(pickled_df_path)
	objects_list = list_objects_generator(triples)

	# generate file list fom folder containing text corpus

	files_list = file_list_gen(args.corpus_path)

	#specify target drive, number of checkpoint intervals

	target_dir = input("Enter path for target directory to write pickled list of enclosed sentences: ")

	num_intervals = int(input("Enter number of checkpoint intervals:"))

	# return sentences with mentions of the object list 

	toy_sentences = enclosed_sentences(files_list, objects_list, nlp, num_intervals, target_dir)

	file_name = target_dir + "enclosed_sentences_final.pkl"

	open_file = open(file_name, "wb")
	pickle.dump(enclosed_sentences, open_file)


	print("Done")
