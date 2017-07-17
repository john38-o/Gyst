#Gyst Outliner
# OUTLINER

import nltk, re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet 
from collections import Counter
# from text_grab import Text_Grab

class Outliner:

	def __init__(self, text, title=None, keywords=[], n_kwd=5, size=0.1):
		# TILTE OF OUTLINE
		self.title = title
		# UNALTERED BODY OF TEXT
		self.text = text
		# OPTIONAL KEYWORDS FOR SIGNIFICANCE SEACRCHING
		self.keywords = keywords
		# EITHER THE NUMBER OF KEYWORDS INPUTED, OR THE NUMBER OF KEYWORDS TO LOOK FOR
		self.n_kwd = n_kwd
		# WHAT PERPORTION OF THEMES TO TOTAL SENTANCES TO LOOK FOR
		self.size = size
	
	'''
	--METHOD-OLOGY--

	'get_Phrases()' REMOVES IRRELEVANT SENTANCES AND SPLITS OTHERS INTO NOUN AND VERB PHRASES.

	'''
	def get_Phrases(self):
		verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'RB', 'RBR', 'RBS', 'TO']
		noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']
		pronoun_tags_2 = ['this', 'that', 'those']
		# self.get_Quotes()

		sentances_ = sent_tokenize(self.text)

#		REMOVE QUESTIONS FROM THE WORKING TEXT BODY
		sentances = []
		for sentance in sentances_:
			if sentance[-1] != '?':
				sentances.append(sentance)

#		BREAKS EACH SENTENCE INTO WORKING CLAUSES -- BY SEMICOLON 
		clause_bag = []
		for sentance in sentances:	
			words = word_tokenize(sentance)
			if ';' in sentance:
				clause_bag.append(words[:words.index(';')])
				clause_bag.append(words[words.index(';') + 1:])
			else:
				clause_bag.append(words)
		self.clause_bag = clause_bag

#		RUNS THROUGH EACH CLAUSE AND SEPARATES THE SENTANCE INTO NOUN PHRASE AND VERB PHRASE HALVES
		halves = []
		for clause in clause_bag:
			words_tagged = nltk.pos_tag(clause)
			removed_extran = False

#			IF ',' FOUND AND THE SENTANCE STARTS WITH A PERPOSITION -- REMOVE THE PHRASE
			for w in words_tagged:
				if w[0] == ','  and (words_tagged[0][1] in ['IN'] or words_tagged[0][0] == ',') and not removed_extran:
						got_noun = False
						got_verb = False
						ix = 0

						while (not got_verb or not got_noun) or ix < len(clause[:clause.index(w[0])]):
							if words_tagged[ix][1] in noun_tags:
								got_noun = True
							if words_tagged[ix][1] in verb_tags:
								got_verb = True
							ix += 1

						if not got_verb or not got_noun:
							clause = clause[clause.index(w[0]) + 1:]	
							removed_extran = True
						else:
							pass
				else:
					pass

#			FIND THE FIRST VERB IN THE CLAUSE
			verb_locs = []
			found_nouns = False
			words_tagged = nltk.pos_tag(clause)
			for i in range(len(words_tagged)):
				if words_tagged[i][1] in noun_tags and words_tagged[i][0] not in ['[', ']', '``', "''"]:
					found_nouns = True
					
				# if words_tagged[i][1] in verb_tags and words_tagged[i][0] != "'s" and i != 0 and not found_nouns:
				# 	for ii in range(len(words_tagged[:i])):
				# 		if words_tagged[ii][0].lower() in pronoun_tags_2:
				# 			found_nouns = True
							
				if words_tagged[i][1] in verb_tags and words_tagged[i][0] != "'s" and i != 0 and found_nouns:
					verb_locs.append(i)
					break

#			IF VERBS ARE FOUND
			if verb_locs != []:
				NP = clause[:verb_locs[0]]
				VP = clause[verb_locs[0]:]
			else:
				pass

			# print(NP, '\n', VP, verb_locs, '\n')
			if NP != [] and VP != []:
				halves.append([NP, VP])

		self.halves = halves
		return halves

	'''
	--METHOD-OLOGY--

	'get_Subject()' RUNS THROUGH EACH NOUN PHRASE AND PULLS OUT THE SENTANCES SUBJECT.

	'''	
	def get_Subject(self):
		halves = self.get_Phrases()

		NP = []
		subjects = []

#		LOADS ALL NOUN PHRASES INTO 'NP'
		for i in range(len(halves)):
			NP.append(halves[i][0])

#		RUNS THROUGH EACH NOUN PHRASE AND EXTRACTS THE SUBJECT
		for np in NP:
			if np != []:

#				REMOVE MEANINGLESS FLUFF WORDS
				words_tagged = nltk.pos_tag(np)
				for i in range(len(words_tagged)):

					if (words_tagged[i][1] in ['EX', 'DT', 'CC', '``', "''"] or words_tagged[i][0].lower() in ['while', 'the', 'you', 'moreover']) and words_tagged[i][0].lower() not in ['this', 'that', 'those']:
						if len(np) > 1 and words_tagged[i][0] in np:
							np.remove(words_tagged[i][0])

					if words_tagged[i][0] == ','  and (words_tagged[0][1] in ['IN'] or words_tagged[0][0] == ','):
						np = np[i + 1:]
					

				if np ==[]:
					for w in words_tagged:
						if words_tagged[0][1] not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VBG', 'JJ', 'JJR', 'JJS']:
							np.append(w[0])
				if np != []:
					optimized = False
					
					while not optimized:
						words_tagged = nltk.pos_tag(np)
						# print(np)
			
						if (words_tagged[0][1] not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VBG', 'JJ', 'JJR', 'JJS'] and len(np) > 1) or words_tagged[0][0] in ['[', ']']:
							# if words_tagged[0][0].lower() not in ['this', 'that', 'those']:
							np.remove(words_tagged[0][0])
							# elif words_tagged[1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
							# 	np.remove(words_tagged[0][0])

						if words_tagged[-1][1] not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$'] or words_tagged[-1][0] in ['[', ']', '``', "''", ','] and len(np) > 1:
							np.remove(words_tagged[-1][0])

						if (words_tagged[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VBG', 'JJ', 'JJR', 'JJS'] and words_tagged[-1][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$'] and words_tagged[0][0] not in ['[', ']'] and words_tagged[-1][0] not in ['[', ']'] ) or len(np) <= 1:
							optimized = True

				if not np or (len(np) == 1 and np[0] in ['[', ']', '``', "''", ',']):
					subjects.append(['--'])
				else:
					subjects.append(np)

# 		CHANGES PRONOUNS TO PREVIOUS SUBJECT, IF NEED BE.
		subs_changed = []
		for i in range(len(subjects)):
			# print(i, len(subs_changed), type(subjects[i]))
			tags = nltk.pos_tag(subjects[i])
			for tag in tags:
				# print(tag)
				if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
					subs_changed.append(subjects[i])
					break
				elif tag[1] in ['PRP', 'PRP$'] and len(subs_changed) > 0:
					subs_changed.append(subs_changed[i-1])
					break
				elif tag[0] in ['this', 'that', 'those'] and tags[tags.index(tag) + 1][1] not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
					subs_changed.append(subs_changed[i-1])
					break
				else:
					subs_changed.append(subjects[i])
					break
				# elif tag[0].lower() in ['this', 'that', 'those']:
				# 	subs_changed.append(subs_changed[i - 1] + halves[i - 1][1])

		# ENSURES EACH SENTANCE HAS A SUBJECT
		if len(subs_changed) != len(halves):
			print('Error with getting Subjects. Got {} sentances and {} subjects'.format(len(halves), len(subs_changed)))
			return None

		else:
#			POPULATES SUPER-LIST OF SUBJECTS AND NOUN/VERB PHRASES
			sub_np_vp = []

			for i in range(len(subs_changed)):
				sub_np_vp.append([subs_changed[i], halves[i][0], halves[i][1]])
			# ANALYSIS ITEM #
			print(subs_changed)
			self.subjects = subs_changed
			self.sub_np_vp = sub_np_vp

			return subs_changed, sub_np_vp


	'''
	--METHOD-OLOGY--

	'Build_Score_Sheet()' SCORES EACH SUBJECT BASED ON ITS FREQUENCY IN THE TEXT IN RELATION TO THE OVERALL SIZE OF THE TEXT. 
						  RETURNS A SORTED LIST OF ALL SUBJECTS AND THEIR SCORE -- [SCORE, SUBJECT]

	EQUATION:
		SCORE = (WORD FREQUENCY / TOTAL WORD COUNT) * 1000

	'''
	def Build_Score_Sheet(self):
		subjects, sub_np_vp = self.get_Subject()	
		all_words = word_tokenize(self.text)

#		POPULATES A 1-D LIST WITH ALL SUBJECTS FOUND
		all_subs = []
		for subject in subjects:
			for word in subject:
				all_subs.append(word)

#		FILTERS SUBJECTS SO THE ALGO ONLY DEALS WITH NOUNS
		filtered_words = []
		noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
# 		LEMMATIZES EACH WORD TO IMPROVE MATCH ACCURACY AND REMOVE DUPLICATES
		lemmatizer = WordNetLemmatizer()
		all_words_lem = [lemmatizer.lemmatize(w).lower() for w in all_words]
		all_subs_lem = [lemmatizer.lemmatize(w).lower() for w in all_subs]

		count = 0
		for w in nltk.pos_tag(all_words_lem):

			if w[1] in noun_tags and w[0] in all_subs_lem:
				filtered_words.append(w[0])

			count += 1

#		POPULATES DICTIONARY WITH EACH UNIQUE SUBJECT AND HOW OFTEN IT OCCURS IN TEXT {WORD: COUNT}
		words_counted = dict(Counter(filtered_words))

#		EACH SUBJECT'S SCORE IS EQUIVILANT TO HOW OFTEN IT OCCURS OVER THE TOTAL NUMBER OF WORDS IN THE TEXT
		words_weighted = []
		for key in words_counted: 
			score = (float(words_counted[key]) / float(count)) * 1000
#			INCREASES SCORE BY 50% IF THE WORD IS A 'KEYWORD'
			if key in self.keywords:
				score *= 1.5
			words_weighted.append([score, key])
		words_weighted = sorted(words_weighted)[::-1]

#		IF NO KEYWORDS ARE GIVEN -- USE THE MOST POPULAR SUBJECTS
		if self.keywords == []:
			for item in words_weighted[:self.n_kwd]:
				self.keywords.append(item[1])

		# ANALYSIS ITEM #
		print('\n', words_weighted, '\n')
		return words_weighted

	'''
	--METHOD-OLOGY--

	'get_Themes()' EXTRACTS THE MOST IMPORTANT SUBJECT-VERB PHRASE PAIRS FROM THE TEXT

	'''	
	def get_Themes(self):
		scored_sents = []
		words_weighted = self.Build_Score_Sheet()
		lemmatizer = WordNetLemmatizer()

#		SCORES AND SORTS EACH SUBJECT-VERB PHRASE PAIR BY ADDING UP EACH NOUNS SCORE
		for item in self.sub_np_vp:
			sentance_score = 0
			for word in item[0] + item[2]:
				for duo in words_weighted:
					if lemmatizer.lemmatize(word.lower()) == duo[1]:
						sentance_score += duo[0]
			scored_sents.append([sentance_score, item[0], item[2]])
		scored_sents_sorted = sorted(scored_sents)[::-1]

#		SAVES THE TOP 'n_themes' MOST IMPORTANT S-VP PAIRS
		n_themes = int(self.size * len(self.halves))
		themes = scored_sents_sorted
		top_subs, top_sents = [], []
		enough_themes = False
		count = 0

#		RUNS UNTIL 'n_themes' THEMES HAVE BEEN LOADED INTO 'top_sents'
		while not enough_themes:
			proxy = themes[count][1], themes[count][2]

			if themes[count][1] not in top_subs:
				# ANALYSIS ITEM #
				print(proxy, '\n\n')
				top_sents.append(proxy)
				top_subs.append(themes[count][1])
			if len(top_sents) >= n_themes:
				enough_themes = True
			count += 1

		self.top_sents = top_sents
		return top_sents


	'''
	--METHOD-OLOGY--

	'get_Points()' RE-EVALUATES EACH NON MAIN-THEME SENTANCE AGAINST EACH MAIN-THEME SENTANCE.
				   MATCHES EACH SUB-POINT WITH THE SENTANCE IT PAIRS BEST WITH -- IF IT CROSSES THE NECCESARY THRESHOLD.
	'''		
	def get_Points(self):
		top_sents = self.get_Themes()
		master_score_dict = {}
		master_score_list = []
		index_1 = 0

#		USES VERB PHRASES AS A WAY TO IDENTIFY WHICH SENTNACES ARE ALREADY MAIN-THEME SENTNACES
		VP_to_aviod = [top_sents[i][1] for i in range(len(top_sents))]

#		RUNS THROUGH EACH NOUN PHRASE AND VERB PHRASE SEPERATELY
		for half in self.sub_np_vp:
			score_dict = {}
			index_2 = 0

			if half[2] not in VP_to_aviod:

				for top_sent_ in top_sents:
					score = 0
					subject = [word.lower() for word in top_sent_[0]]
					predicate = [word.lower() for word in top_sent_[1]]

#					COMPARES WORDS IN SUBECT OF SENTANCE AGAINST EACH MAIN-THEME
					words_tagged = nltk.pos_tag(half[0])
					for word in (words_tagged):

						if word[1] in ['NNP', 'NNPS'] and word[0].lower() in subject:
							score += 2
						if word[1] in ['NN', 'NNS'] and word[0].lower() in subject:
							score += 2

						if word[1] in ['NN', 'NNS', 'NNP', 'NNPS'] and word[0].lower() in predicate:
							score += 1.5

					score = float(score)

#					COMPARES WORDS IN PREDICATE OF SENTANCE AGAINST EACH MAIN-THEME
					words_tagged = nltk.pos_tag(half[2])
					for word in (words_tagged):

						if word[1] in ['NNP', 'NNPS'] and word[0].lower() in subject:
							score += 1.7
						if word[1] in ['NN', 'NNS'] and word[0].lower() in subject:
							score += 1.3

						if word[1] in ['NN', 'NNS', 'NNP', 'NNPS'] and word[0].lower() in predicate:
							score += 1

#					FORMATS THE SCORE, DICTIONARY AND LIST -- SCORE DICT {MAIN-THEME ID: SCORE} -- MASTER LIST FOR AVERAGEING LATER
					score = float(score) * 1000.0
					score_dict[index_2] = score
					master_score_list.append(score)

					index_2 += 1

#			SCORES MAIN-THEMES AGAINST MAIN-THEMES AS 0.0 
			else:
				for top_sent in top_sents:

					score = 0.0
					score_dict[index_2] = score
					master_score_list.append(score)
					index_2 += 1

#			{LOCATION OF SENTNACE: {MAIN-THEME ID: SCORE}}
			master_score_dict[index_1] = score_dict
			index_1 += 1

		# ANALYSIS ITEM #
		print('\n', master_score_dict, '\n')	

#		CALCULATES THRESHOLD SCORES NEED TO CROSS
		master_score_list = [n for n in master_score_list if n != 0.0]
		master_score_list = np.array(master_score_list)
		score_std = np.std(master_score_list)
		avg_score = np.mean(master_score_list)
		threshold = avg_score + (1 * score_std)

		# ANALYSIS ITEM #
		print('Average:', avg_score)
		print('Threshold:', threshold)

#		ASSIGNS THRESHOLD-CROSSING LOCATIONS OF SUB-POINTS IN TEXT TO A MAIN-THEME IT SCORED HIGHEST W/ IN THE 2-D ARRAY 'point_list'
		point_list = []
		for k in master_score_dict[0]:
			point_list.append([])

		for key in master_score_dict:
			scores = []

			for k in master_score_dict[key]:
				scores.append(master_score_dict[key][k])

			for i in range(len(scores)):
				if scores[i] >= threshold and scores[i] == sorted(scores)[::-1][0]:
					point_list[i].append(key)

#		REFINES 'point_list' TO INCLUDE ONLY THE ABOVE-AVERAGE ANNECDOTES
		for i in range(len(point_list)):
			scores_in_list = []

			for sent_loc in point_list[i]:
				scores_in_list.append(master_score_dict[sent_loc][i])

			if scores_in_list != []:
				scores_in_list = [n for n in scores_in_list if n != 0.0]
				scores_in_list_np = np.array(scores_in_list)
				avg_score_per_list = np.mean(scores_in_list_np)
				# print(i, 'Scores:', scores_in_list, '\n', 'Average:', avg_score_per_list)

				proxy = []
				for sent_loc in point_list[i]:
					if master_score_dict[sent_loc][i] >= avg_score_per_list:
						proxy.append(sent_loc)
				point_list[i] = proxy

			else:
				point_list[i] = point_list[i]

		# ANALYSIS ITEM #
		print('\n', point_list, '\n')

#		POPULATES 'top_sents_w_points' CONTAINING MAIN-THEME'S RANK (0-n) AND CONTENTS, FOLLOWED BY ITS ANNECDOTES
		top_sents_w_points = []
		for i in range(len(top_sents)):
			sent = [top_sents[i][0], top_sents[i][1]]
			sentances = []

			# ANALYSIS ITEM #
			print('\n', i, sent, '\n')

			for location in point_list[i]:
				sentances.append([self.sub_np_vp[location][0], self.sub_np_vp[location][2]])

				# ANALYSIS ITEM #
				print('Annecdote:', [self.sub_np_vp[location][0], self.sub_np_vp[location][2]],
					  'Score:', master_score_dict[location][i])

			top_sents_w_points.append([i, sent, sentances])
		

		return top_sents_w_points

	'''
	--METHOD-OLOGY--

	'get_Quotes()' RUNS THROUGH self.text AND PULLS OUT THE RELEVANT QUOTES, THEIR SPEAKERS AND SUBJECTS.

	'''
	def get_Quotes(self):
		words = word_tokenize(self.text)

#		FOR EACH WORD: SAVE LOCATION OF THE FIRST START QUOTATION IT FINDS AND THE FOLLOWING END QUOTATION. TAKES THE 		  SLICE OF WORDS BETWEEN THE QUOTE LOCATIONS AND APPENDS THEM INTO 'quotes_dirty'
		start_q = None
		end_q = None
		quotes_dirty = []
		for i in range(len(words)):
			
			if words[i] == '``':
				start_q = i

			if words[i] == "''" and start_q != None:
				end_q = i

				quotes_dirty.append([words[start_q:(end_q + 1)], [start_q, end_q]])
				start_q = None
				end_q = None


#		IF THE QUOTE FOUND IS TOO SMALL, REMOVE IT
# 		IF THE QUOTE FOUND HAS PUNCTUATION AT THE END, REMOVE PUNCTUATION
		quotes_clean = []
		for quote in quotes_dirty:
			if len(quote[0]) > 5:
				if quote[0][-2] not in ['.', ',']:
					quotes_clean.append(quote)
				else:
					quote[0] = quote[0][:-2]
					quote[0].append("''")
					quotes_clean.append(quote)
				# print(quote, '\n')


# 		FIND THE SPEAKERS OF THE QUOTES
		speakers = []
		for quote in quotes_clean:
			found_speaker = False
			speaker = []
#			GRABS THE 'words' INDEX OF THE START AND END OF THE QUOTE FOR THE QUOTE IN QUESTION 
			start_q = quote[1][0]
			end_q = quote[1][1]

#			ITERATES ACROSS 5 WORDS LEFT FROM THE START OF THE QUOTE
			count = 1
			while (not found_speaker and count <= 5):
				word_to_check = words[start_q - count]

#				IF THE WORD IS A PERIOD, SINCE NO NAMES WERE FOUND, BREAK THE LOOP
				if word_to_check == '.':
					break
#				IF THE WORD'S FRIST LETTER IS UPPERCASE -- POTENTIAL NAME
				elif word_to_check[0].isupper():
					speaker.insert(0, word_to_check)
#				IF THE WORD'S FRIST LETTER IS LOWERCASE AND NAMES HAVE BEEN ADDED -- ALL NAMES HAVE BEEN FOUND
				elif len(speaker) > 0:
					found_speaker = True
				count += 1

			if end_q < len(words) - 6:
				count = 1

#				IF NO NAMES WERE FOUND ON THE LEFT OF THE QUOTE -- LOOK 5 WORDS TO THE LEFT
				while (not found_speaker and count <= 5):
					word_to_check = words[end_q + count]

#					IF THE WORD IS A PERIOD, OR THE WORD BEFORE THE QUOTE IS A PERIOD
					if words[end_q - 1] == '.' or word_to_check == '.':
						break
#					IF THE WORD'S FRIST LETTER IS UPPERCASE -- POTENTIAL NAME
					elif word_to_check[0].isupper():
						speaker.append(word_to_check)
#					IF THE WORD'S FRIST LETTER IS LOWERCASE AND NAMES HAVE BEEN ADDED -- ALL NAMES HAVE BEEN FOUND
					elif len(speaker) > 0:
						found_speaker = True
					count += 1

			speakers.append(speaker)

#		FIND SPEAKERS FOR THE QUOTES THAT WEREN'T IDENTIFIED ABOVE
		speakers_filled = []
		for i in range(len(speakers)):

#			IF THE EMPTY NAME IS NOT THE FIRST IN THE LIST, USE THE PREVIOUS NAME
			if speakers[i] == [] and i > 0:
				speakers_filled.append(speakers[i - 1])
#			IF THE EMPTY NAME IS THE FIRST IN THE LIST, USE THE FOLLOWING NAME
			elif speakers[i] == [] and i == 0:
				speakers_filled.append(speakers[i + 1])
#			IF THE NAME ISN'T EMPTY SEND IT BACK THROUGH
			else:
				speakers_filled.append(speakers[i])

#		FIND THE FULL NAME OF EACH SPEAKER
		speakers_first_last = []
		for spkr in speakers_filled:

#			IF THE SPEAKER ONLY HOLDS ONE NAME -- LOOK FOR MORE
			if len(spkr) == 1:
				speaker = spkr
				got_name = False

#				GOES THROUGH ALL THE WORDS IN TEXT AND FIND A MATCH WITH THE NAME
				for i in range(len(words)):

					if words[i] == spkr[0]:

						count = 1
#						LOOK TO THE LEFT OF THE WORD
						while not got_name and count <= 3:

							if not words[i - count][0].isupper():
								break
							elif words[i - count][0].isupper() or nltk.pos_tag(words[i - count]) in ['PRP', 'PRP$']:
								speaker.insert(0, words[i - count])
							elif len(speaker) > 1:
								got_name = True
							count += 1

						count = 1
#						IF NONE TO THE LEFT, LOOK TO THE RIGHT
						while not got_name and count <= 3:

							if not words[i + count][0].isupper():
								break
							elif words[i + count][0].isupper()  or nltk.pos_tag(words[i - count]) in ['PRP', 'PRP$']:
								speaker.append(words[i + count])
							elif len(speaker) > 1:
								got_name = True
							count += 1

				speakers_first_last.append(speaker)

			elif len(spkr) > 1:
				speakers_first_last.append(spkr)
			else:
				quotes_clean.remove(quotes_clean[speakers_filled.index(spkr)])

#		MAKE SURE THERE IS ONE NAME FOR EACH QUOTE
		quotes = None
		if len(speakers_first_last) == len(quotes_clean):

#			FORMAT THE LIST OF QUOTES BY QUOTE, LOCATION AND SPEAKER
			finished_quotes = []
			for i in range(len(quotes_clean)):
				finished_quotes.append([quotes_clean[i], speakers_first_last[i]])
			quotes = finished_quotes

		else:
			print("Number of quotes and speakers don't match. \nQuotes: {}  Speakers: {}".format(len(quotes_clean), len(speakers_first_last)))

#		FINDS LOCATION OF RELEVANT SUBJECTS IN THE BODY OF TEXT
		subjects = self.subjects

		sub_locs = []
		for i in range(len(words)):
			if words[i] == '``' and "''" not in [words[i + 2], words[i + 3], words[i + 4]]:
				start_q = i
				period_loc = []
				found_first_period = False
				found_second_period = False
				count = 1

				while not found_first_period:

					if words[i - count] == '.':
						found_first_period = True
						period_loc.append(i - count)
					count += 1

				while not found_second_period:

					if words[i - count] == '.':
						found_second_period = True
						period_loc.insert(0, i - count)
					count += 1

				sub_locs.append(period_loc)
		
#		FINDS SUBJECT OF QUOTES AND REPLACES PRONOUNS
		quote_subs = []
		for loc in sub_locs:
			sentance = words[loc[0] + 1:loc[1] + 1]
			got_sub = False
			i = 0

			while not got_sub and i < len(self.halves):
				if all(word in sentance for word in self.halves[i][1]):
					quote_subs.append(subjects[i])
					got_sub = True
				i += 1

			if not got_sub:
				quote_subs.append(quote_subs[len(quote_subs) - 1])
				got_sub = True
				
		quote_w_subs = []
		for ix in range(len(quotes)):
			words_tagged = nltk.pos_tag(quotes[ix][0][0])
			proxy = []
			found_pro = False
			found_noun = False

			for word in words_tagged:
				if word[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'RB', 'RBR', 'RBS', 'TO'] and word[0] not in ['[', ']']:
					for w in words_tagged[:words_tagged.index(word)]:
						if w[1] in ['NN', 'NNS', 'NNP', 'NNPS'] and w[0] not in ['[', ']']:
							found_noun = True
					if not found_noun:
						for w in words_tagged[:words_tagged.index(word)]:
							if w[1] in ['PRP', 'PRP$']:
								found_pro = True
			if found_pro:
				changed_pro = False
				for word in words_tagged:
					if word[1] in ['PRP', 'PRP$'] and not changed_pro:
						proxy.extend(quote_subs[ix])
						proxy.insert(1, '[')
						proxy.append(']')
						changed_pro = True
					elif word[0] not in ['[', ']']:
						proxy.append(word[0])

				quote_w_subs.append([proxy, quotes[ix][1]])
			else:
				quote_w_subs.append([quotes[ix][0][0], quotes[ix][1]])

		for q in quote_w_subs:
			# ANALYSIS ITEM #
			print(q, '\n')
		return quote_w_subs

	def format(self):
		print('THE FOLLOWING IS FOR DEBUGGING, OUTPUT IN TEXTFILE.\n\n\n')
		unformatted = self.get_Points()
		quotes = self.get_Quotes()
		title = self.title


		string_to_print = ''
	
		for point in unformatted:
			string_to_print += str(point[0] + 1) + '.) ' + '[' + ' '.join(point[1][0]) + '] ' + ' '.join(point[1][1]) + '\n'
			print('\n\nPoint', point)

			if point[2] != []:
				for sub_pt in point[2]:
					string_to_print += '–– ' + '[' + ' '.join(sub_pt[0]) + '] ' + ' '.join(sub_pt[1]) + '\n'

			string_to_print += '\n'


		string_to_print += '\nImportant Quotes:\n\n'
		for quote in quotes:
			string_to_print += ' '.join(quote[0]) + '[' + ' '.join(quote[1]) + ']\n\n'
		if quotes == []:
			string_to_print += '//none//'

		print('\nTo TXT:\n\n', string_to_print)

		out_file = open('gyst_output.txt', 'w')
		out_file.write(string_to_print)


	def run(self):
		self.format()

file_name = input('FileName (w/o ext): ')
file_name += '.txt'
file = open(file_name, 'r')
text = file.read()
outliner = Outliner(text)
outliner.run()