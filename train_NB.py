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

