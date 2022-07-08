
from collections import defaultdict
import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

class KeywordExtractor:
    def __init__(self, docs) -> None:
        self.docs = docs
        self.tf = self.get_tf()
        self.word_count_df = pd.DataFrame({'count':(self.tf > 0).sum().sort_values()})

    def get_tf(self,):
        vect = CountVectorizer()
        document_term_matrix = vect.fit_transform(self.docs)
        tf = pd.DataFrame(document_term_matrix.toarray(), columns=vect.get_feature_names()) 
        return tf

    def extract_keywords(self,):
        raise NotImplementedError("extract_keywords 메소드를 구현하여야 합니다.")

    def get_keywords(self, duplicate_limit=0.1, num_keywords=500) -> set:
        word_set = set(self.word_count_df.query('count>1').query(f'count<{duplicate_limit*len(self.docs)}').index.tolist())
        filtered_kw = set()
        count = 0
        for w, s in sorted(self.keywords.items(), key=lambda item: item[1])[::-1]:
            if w in word_set:
                filtered_kw.add(w)
                count += 1
            if count == num_keywords:
                break
        return filtered_kw
        
class KeyBertExtractor(KeywordExtractor):
    def __init__(self, docs) -> None:
        super().__init__(docs)
        self.n_gram_range = (1, 1)
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
