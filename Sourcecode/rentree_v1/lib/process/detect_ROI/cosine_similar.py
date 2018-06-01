# -*- coding: utf-8 -*-
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import re

#nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def find_lines_similar(corpus, text):
	line_similar=""
	score=0

	#lines= strip_accents(corpus).split('\n')
	#print lines
	for line in corpus:
		sc=cosine_sim(strip_accents(line), text)
		#print sc, line
		print levenshteinDistance(strip_accents(line), text)
		if sc>score:
			line_similar=line
			score=sc

	return line_similar, score


def convert_corpus_bagsofwords(corpus):
	punctuation_pattern = ' |\.$|\. |\n| |, |\/|\(|\)|\'|\"|\!|\?|\+'
	text = "This article is talking about vue-router. And also _.js."
	ltext = strip_accents(corpus).lower()
	w=ltext.splitlines()
	wtext = [w for w in re.split(punctuation_pattern, ltext) if w]
	return wtext

def check_filled_box(img_file, str_check,threshold):
	#text=read_tesseract_file(img_file)
	text = pytesseract.image_to_string(Image.open(img_file), config='-psm 4')
	print text
	t=strip_accents(str_check)
	text=strip_accents(text)
	corpus = [line.strip() for line in text.split("\n")]
	print corpus
	line_similar,score=find_lines_similar(corpus,t)
	print "line similar:", line_similar
	print "score:", score
	if score<threshold and levenshteinDistance(strip_accents(line_similar), str_check)>=4:
		return True
	return False


if __name__ == '__main__':	
	img_file="test6.png"
	t1="Fait a , le Signature"
	threshold=0.8
	t2="Fait a , le"
	if check_filled_box(img_file, t1, threshold) and check_filled_box(img_file, t2, threshold):
		print "change at", t1


			

# print cosine_sim('a little bird', 'a little bird')
# print cosine_sim('a little bird', 'a little bird chirps')
# print cosine_sim('Fait à , le Signature', 'Fait a , le Signature')





