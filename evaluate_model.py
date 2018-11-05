from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(cleaned_labels,test_size=0.33,random_state=42)
term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)

#compute prior and likelihood based on training set
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train,label_index,smoothing)

#predict the posterior of the test dataset
term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test,prior,likelihood)

#evaluate the model based on number of correct prediction
correct=0.0
for pred,actual in zip(posterior,Y_test):
    if actual==1:
        if pred[1]>=0.5:
            correct+=1
        elif pred[0]>0.5:
            correct+=1

print('The accuracy on {0} testing samples is: {1:.1f}%' .format(len(Y_test), correct/len(Y_test)*100))