
import pandas as pd
import numpy as np
import nltk
from collections import OrderedDict, defaultdict

from keyword_extractor import KeyBertExtractor, TFIDFExtractor, TopicRankExtractor, \
                              TextRankExtractor, KeyBertEmbeddingExtractor

def get_keyword_co_occurrence_matrix(nid_arr_dict):
    nid_arr_dict = OrderedDict(sorted(nid_arr_dict.items()))
    keyword_matrix = np.array([ item for k, item in nid_arr_dict.items() ])
    keyword_co_occurrence_matrix = np.matmul(keyword_matrix, keyword_matrix.T)
    
    return  keyword_co_occurrence_matrix


def convert_doc_array(item_docs, user_docs, kw_df):
    
    def get_keyword_vector(text:str, keywords_index_dict:dict):
        n = len(keywords_index_dict)
        one_hot_vector = [0]*n
        words = text.split()
        for w in words:
            i = keywords_index_dict.get(w)
            if i != None:
                one_hot_vector[i] = 1

        return one_hot_vector
    

    keywords_index_dict = {}
    kw_id = 0 
    kw_set = set(kw_df['keyword'])
    for kw in kw_set:
        if kw not in keywords_index_dict:
            keywords_index_dict[kw] = kw_id
            kw_id += 1

    nid_arr_dict = {}
    print(len(item_docs), len(user_docs))
    for i, d in zip(item_docs.index, item_docs.values):
        nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)

    for i, d in zip(user_docs.index, user_docs.values):
        nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)
    
    return nid_arr_dict


if __name__=='__main__':
    # load docs 
    keyword_extractors = {
        'text_rank' : TextRankExtractor,
        'tfidf' : TFIDFExtractor,
        'topic_rank' : TopicRankExtractor,
        'keybert' : KeyBertExtractor,
    }

    keyword_extraction_method='keybert'
    for data_name in ['game', 'music', 'book', 'office', 'sports']:
        train_df = pd.read_csv(f'data/{data_name}/{data_name}_train.csv')
        valid_df = pd.read_csv(f'data/{data_name}/{data_name}_valid.csv')
        test_df = pd.read_csv(f'data/{data_name}/{data_name}_test.csv')
        # df = pd.concat([train_df, valid_df, test_df])
        df = pd.concat([train_df, valid_df,])
        df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        item_docs = df.groupby('item_id')['text_clean'].apply(lambda x: ' '.join(x))
        user_docs = df.groupby('user_id')['text_clean'].apply(lambda x: ' '.join(x))
        docs = item_docs.to_list() + user_docs.to_list()

        keyword_extractor_class = keyword_extractors.get(keyword_extraction_method)
        keyword_extractor = keyword_extractor_class(docs, n_gram_range=(1,1))
        keyword_extractor.extract_keywords(top_n=5)
        keywords = keyword_extractor.get_keywords(duplicate_limit=0.1, num_keywords=512)
        print(list(keywords)[:40])
        kw_df = pd.DataFrame({'keyword':list(keywords)})
        kw_df.to_csv(f'data/{data_name}/{keyword_extraction_method}_keywords.csv') 
        
        nid_arr_dict = convert_doc_array(item_docs, user_docs, kw_df)
        cooc_matrix = get_keyword_co_occurrence_matrix(nid_arr_dict)

        with open(f'data/{data_name}/{keyword_extraction_method}_ttf_idkf_cooc_matrix.npy', 'wb') as f:
            np.save(f, cooc_matrix)


        # Extract Bert embeddings
        # if keyword_extraction_method == 'bert_embedding':
        # keybert_embedding_extractor = KeyBertEmbeddingExtractor(item_docs)
        # item_embeddings = keybert_embedding_extractor.extract_embeddings()
        # keybert_embedding_extractor = KeyBertEmbeddingExtractor(user_docs)
        # user_embeddings = keybert_embedding_extractor.extract_embeddings()
        # print(type(item_embeddings))
        # print(item_embeddings.shape)

        # embeddings = np.concatenate([user_embeddings, item_embeddings], axis=0)
        # print(embeddings.shape)
        # with open(f'data/{data_name}/doc_embeddings.npy', 'wb') as f:
        #     np.save(f, embeddings)
