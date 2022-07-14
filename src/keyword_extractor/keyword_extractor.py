
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from topicrank import TopicRank
from tqdm import tqdm

class KeywordExtractor:
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        self.n_gram_range = n_gram_range
        self.docs = docs
        self.tf = self.get_tf()
        self.word_count_df = pd.DataFrame({'count':(self.tf > 0).sum().sort_values()})

    def get_tf(self,):
        self.vect = CountVectorizer(ngram_range=self.n_gram_range,)
        document_term_matrix = self.vect.fit_transform(self.docs)
        tf = pd.DataFrame(document_term_matrix.toarray(), columns=self.vect.get_feature_names()) 
        return tf

    def extract_keywords(self,):
        raise NotImplementedError("extract_keywords 메소드를 구현하여야 합니다.")

    def get_keywords(self, duplicate_limit=0.1, num_keywords=500) -> set:
        word_set = set(self.word_count_df.query('count>1').query(f'count<{duplicate_limit*len(self.docs)}').index.tolist())
        # print(list(word_set)[:10])
        filtered_kw = set()
        count = 0
        for w, s in sorted(self.keywords.items(), key=lambda item: item[1])[::-1]:
            if w in word_set:
                filtered_kw.add(w)
                count += 1
            if count == num_keywords:
                break
        return filtered_kw

class TFIDFExtractor(KeywordExtractor):
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        super().__init__(docs, n_gram_range)
        self.stop_words = "english"
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, smooth_idf=True, use_idf=True)
        self.vectorizer.fit_transform(self.docs)
        self.feature_names = self.vectorizer.get_feature_names()
    
    def _sort_coo(self, coo_matrix):
        """Sort a dict with highest score"""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        
        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results

    def extract_keywords(self, top_n=5) -> None:
        kws = defaultdict(int)
        for doc in tqdm(self.docs):
            tf_idf_vector = self.vectorizer.transform([doc])
            sorted_items=self._sort_coo(tf_idf_vector.tocoo())
            keywords=self._extract_topn_from_vector(self.feature_names, sorted_items, top_n)
            for kw, s in keywords.items():
                kws[kw] += s     
        self.keywords = kws 

        
class TopicRankExtractor(KeywordExtractor):
    def extract_keywords(self, top_n=5):
        kws = defaultdict(int)
        for doc in tqdm(self.docs):
            tr = TopicRank(doc)
            keywords=tr.get_top_n(n=top_n)
            for kw in keywords:
                kws[kw] += 1     
        self.keywords = kws 

from summa import keywords

class TextRankExtractor(KeywordExtractor):
    def extract_keywords(self, top_n=5):
        kws = defaultdict(int)
        for doc in tqdm(self.docs):
            keyword_list=keywords.keywords(doc).split('\n')[:top_n]
            for kw in keyword_list:
                kws[kw] += 1     
        self.keywords = kws 


class KeyBertExtractor(KeywordExtractor):
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        super().__init__(docs, n_gram_range)
        self.stop_words = "english"
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    def extract_keywords(self, top_n=5) -> None:
        kws = defaultdict(int)
        for doc in tqdm(self.docs):
            try:
                count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words).fit([doc])
            except:
                continue
            candidates = count.get_feature_names_out()

            doc_embedding = self.model.encode([doc])
            candidate_embeddings = self.model.encode(candidates)

            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
            for kw in keywords:
                kws[kw] += 1
        
        self.keywords = kws 
