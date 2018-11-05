import glob
import os


#looking at the non-spam and a spam email where the dataset is in a folder. The dataset contains a folder of spam and
#non spam text files

#looking at the content of the emails
file_path='path/to/the/folder/non_spam.txt'
with open(file_path,'r') as infile:
    non_spam=infile.read()
print(non_spam)

file_path='path/to/the/folder/spam.txt'
with open(file_path,'r') as infile:
    spam=infile.read()
print(spam)


#finding all .txt files and creating variables for the text data and labels
emails=[]
labels=[]
file_path='path/to/spam/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
  with open(filename,'r',encoding= "ISO-8859-1") as infile:
    emails.append(infile.read())
    labels.append(0)

file_path='path/to/non_spam/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
  with open(filename,'r',encoding= "ISO-8859-1") as infile:
    emails.append(infile.read())
    labels.append(0)

#text cleaning
def clean_text(docs):
    cleaned_docs=[]
    for doc in docs:
        cleaned_docs.append(''.join([lemmatizer(word.lower())
                            for word in doc.split()
                            if letters_only (word)
                             and word not in all_names]))
    return cleaned_docs

cleaned_emails=clean_text(emails)

#extracting features which is the term frequencies
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words="english",max_features=500)
term_docs=cv.fit_transform(cleaned_emails)

feature_names=cv.get_feature_names()
feature_mapping=cv.vocabulary_

#####training the naive bayes###
from collections import defaultdict

def get_label_index(labels):
 label_index=defaultdict(list)
 for index,label in enumerate(labels):
     label_index[label].append(index)
 return label_index


#calculate prior
def get_prior(label_index):
  prior={label: len(index) for label, index in label_index.iteritems()}
  total_count=sum(prior.values())
  for label in prior:
      prior[label]/=float(total_count)
  return prior

#calculate likelihood
import numpy as np
def get_likelihood(term_doc_matrix, label_index,smoothing):
    likelihood={}
    for label,index in label_index.iteritems():
        likelihood[label]=term_doc_matrix[index,:].sum(axis=0)+smoothing
        likelihood[label]=np.asarray(likelihood[label]) [0]
        total_count=likelihood[label].sum()
        likelihood[label]= likelihood[label]/float(total_count)
    return likelihood

def get_posterior(term_doc_matrix,prior,likelihood):
    num_docs=term_doc_matrix.shape[0]
    posteriors=[]
    for i in range(num_docs):
        posterior={key: np.log(prior_label) for key,prior_label in prior.iteritems()}

        for label,likelihood_label in likelihood.iteritems():
            term_doc_vector=term_doc_matrix.get_row(i)
            counts=term_doc_vector.data
            indices=term_doc_vector.indices

            for count,index in zip(counts,indices):
                posterior[label]+=np.log(likelihood_label[index])*count

        min_log_posterior=min(posterior.values())
        for label in posterior:
            try:
                posterior[label]=np.exp(posterior[label]-min_log_posterior)
            except:
                posterior[label]=float('inf')
        sum_posterior=sum(posterior.values())
        for label in posterior:
            if posterior[label]==float('inf'):
                posterior[label]=1.0
            else:
                posterior[label]/= sum_posterior
        posteriors.append(posterior.copy())
        return posteriors

