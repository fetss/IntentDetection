import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

def get_data():
    df_command = pd.read_csv('LearnData/seq.in', names = ['Command'])
    df_label = pd.read_csv('LearnData/label', names = ['Label'])
    df_command = df_command[pd.notnull(df_command)]
    df_label = df_label[pd.notnull(df_label)]
    df = pd.concat([df_command, df_label], axis = 1)
    #df = df.sample(frac=1).reset_index(drop=True)
    return df

predefined_token = ['phần trăm']
                    #'kiểm tra',
                    #'có thể',
                    #'làm ơn',
                    #'trạng thái',
                    #'cài đặt']
def custom_tokenizer(string):
    #for token in predefined_token:
    #    string = string.replace(token, token.replace(' ', '_'))
    string = string.split()
    #string = [s if not s.isnumeric() else 'n_u' for s in string]
    string = [s for s in string if not s.isnumeric()]
    return string

def naive_algo():
    data = get_data()

    tfidf_vect = TfidfVectorizer(use_idf=False,
                                 binary=True,
                                 sublinear_tf=True,
                                 min_df=3,
                                 ngram_range=(1, 2),
                                 tokenizer=custom_tokenizer,
                                 max_features=350,
                                 stop_words=['và', 'bạn', 'tôi', 'mình', 'giúp', 'có'])

    #print(list(map(tfidf_vect.build_tokenizer(),data['Command'])))

    X = tfidf_vect.fit_transform(data['Command'])
    clf = MultinomialNB(alpha=0.9, fit_prior=False).fit(X, data['Label'])

    return clf,tfidf_vect

def test():
    clf, tfidf_vect = naive_algo()

    df_test_command = pd.read_csv('TestData/seq.in', names=['Command'])
    df_test_label = pd.read_csv('TestData/label', names=['Label'])
    intent = clf.predict(tfidf_vect.transform(df_test_command['Command']))
    probability = clf.predict_proba(tfidf_vect.transform(df_test_command['Command']))
    metric = metrics.accuracy_score(intent, df_test_label['Label'])

    #for i in range(len(df_test_label['Label'])):
    #    print(intent[i], df_test_label['Label'][i], probability[i])

    return metric

def random_test():
    clf, tfidf_vect = naive_algo()

    df_test_command = pd.read_csv('TestData/seq.in', names=['Command'])
    df_test_label = pd.read_csv('TestData/label', names=['Label'])
    df = pd.concat([df_test_command, df_test_label], axis=1)
    df = df.sample(frac=0.2).reset_index(drop=True)

    intent = clf.predict(tfidf_vect.transform(df['Command']))

    metric = metrics.accuracy_score(intent, df['Label'])
    return metric

if __name__ == '__main__':
    metric = test()
    print(metric)