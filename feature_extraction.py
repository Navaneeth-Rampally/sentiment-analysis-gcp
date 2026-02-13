import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureExtractor:
    """ Feature extraction methods using BOW and TF-IDF"""

    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None

    def fit_transform(self, documents):       
        """ Fit and transform document to feature vectors""" 
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8
            )
            
        # FIX: Apply fit_transform for BOTH methods here
        feature_matrix = self.vectorizer.fit_transform(documents)
        return feature_matrix

    # FIX: Renamed from transform_ to transform (removed underscore)
    def transform(self, documents):
        """ Transform new documents using the appropriate vectorizer"""
        if self.vectorizer is None:
            raise ValueError("The vectorizer has not been fitted yet.")
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        """ Get feature names from the vectorizer"""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()

if __name__ == "__main__":
    sample_docs = [
        "This movie is too booring!.", 
        "Amazing movie to watch and i recommend this to everyone who like watching movies.",
        "Terrible movie.",
        "The lead actor playes an amazing role which makes it interesting to watch this movie."
    ]
    
    # Test BOW
    print("Testing Bag of Words:")
    bow = FeatureExtractor(method='bow', max_features=50)
    bow_matrix = bow.fit_transform(sample_docs)
    print(bow.get_feature_names()[:10])

    # Test TF-IDF
    print("\nTesting TF-IDF:")
    tfidf = FeatureExtractor(method='tfidf', max_features=50)
    tfidf_matrix = tfidf.fit_transform(sample_docs)
    print(tfidf.get_feature_names()[:10])