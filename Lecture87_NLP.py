import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
stopword = stopwords.words('english') 
from nltk import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

messages = [line.rstrip() for line in open('/Users/ifrahkhanyaree/Desktop/Kurzarbeit/Python_Pontilla_Udemy/smsspamcollection/SMSSpamCollection')]

for message_no, message in enumerate(messages[:10]):
	message_no, message
	'\n'

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['labels','message',])

messages['length'] = messages['message'].apply(len)


messages['length'].plot(bins=50,kind='hist', xlim=300)


messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length', by='labels', bins=50,figsize=(10,4))

mess = 'Sample message! Notice it has punctuation!'
nopunc = [char for char in mess if char not in string.punctuation]
nopunc = ''.join(nopunc)

clean_message = [word for word in nopunc.split() if word.lower() not in stopword]
#print(clean_message)


def text_process(mess):
	nopunc = [char for char in mess if char not in string.punctuation]
	nopunc = ''.join(nopunc)
	return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].apply(text_process)


bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer.fit(messages['message'])
messages_bow = bow_transformer.transform(messages['message'])
message4=messages['message'][3]
bow4 = bow_transformer.transform([message4])

#print ('Shape of Sparse Matrix: ', messages_bow.shape)
#print ('Amount of Non-Zero occurences: ', messages_bow.nnz)
#print ('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))


tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf = tfidf_transformer.transform(messages_bow)
tfidf_onemess = tfidf_transformer.transform(bow4)

spam_detect_model = MultinomialNB().fit(tfidf,messages['labels'])
allpredict = spam_detect_model.predict(tfidf)

#print('Predicted:',spam_detect_model.predict(tfidf_onemess))



#print (classification_report(messages['label'], allpredict))

message_train, message_test, label_train, label_test = train_test_split(messages['message'],messages['labels'],test_size=0.2)


pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),
	('tfidf',TfidfTransformer()),
	('classifier',MultinomialNB())])

pipeline.fit(message_train,label_train)
predictions = pipeline(message_test)
print(classification_report(predictions,label_test))

