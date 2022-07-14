
import pandas as pd
import numpy as np
import nltk
from collections import defaultdict

from keyword_extractor import KeyBertExtractor, TFIDFExtractor, TopicRankExtractor, TextRankExtractor


# load yaml configs 

## read yaml list 

## load yaml files 

## build configs 

# extract keywords 
# for cfg in configs:
#     ## config 에 따라 kw extractor 생성
#     kw_extractor = KeywordExtractor(type=type)
#     keywords = kw_extractor.extract_keywords()

    ## save keywords 
from collections import OrderedDict

def get_keyword_co_occurrence_matrix(nid_arr_dict):
    nid_arr_dict = OrderedDict(sorted(nid_arr_dict.items()))
    keyword_matrix = np.array([ item for k, item in nid_arr_dict.items() ])
    keyword_co_occurrence_matrix = np.matmul(keyword_matrix, keyword_matrix.T)
    
    return  keyword_co_occurrence_matrix


def convert_doc_array(item_docs, user_docs, item_kw_df, user_kw_df):
    
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
    kw_set = set(item_kw_df['keyword'])
    kw_set = kw_set.union(set(user_kw_df['keyword']))
    for kw in kw_set:
        if kw not in keywords_index_dict:
            keywords_index_dict[kw] = kw_id
            kw_id += 1

    nid_arr_dict = {}
    print(max(item_docs.index), max(user_docs.index))
    for i, d in zip(item_docs.index, item_docs.values):
        nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)

    for i, d in zip(user_docs.index, user_docs.values):
        nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)
    
    return nid_arr_dict


if __name__=='__main__':
    # load docs 
    keyword_extractors = {
        'text_rank' : KeyBertExtractor,
        'tfidf' : TFIDFExtractor,
        'topic_rank' : TopicRankExtractor,
        'keybert' : KeyBertExtractor,
    }

    keyword_extraction_method='text_rank'
    for data_name in ['toy', 'sports', 'game', 'office',]:
        train_df = pd.read_csv(f'data/{data_name}/{data_name}_train.csv')
        valid_df = pd.read_csv(f'data/{data_name}/{data_name}_valid.csv')
        test_df = pd.read_csv(f'data/{data_name}/{data_name}_test.csv')
        # df = pd.concat([train_df, valid_df, test_df])
        df = pd.concat([train_df, valid_df,])
        df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        # item docs
        item_docs = df.groupby('item_id')['text_clean'].apply(lambda x: ' '.join(x))

        keyword_extractor_class = keyword_extractors.get(keyword_extraction_method)
        keyword_extractor = keyword_extractor_class(item_docs, n_gram_range=(1,1))
        keyword_extractor.extract_keywords(top_n=5)
        keywords = keyword_extractor.get_keywords(duplicate_limit=0.1, num_keywords=500)
        print(list(keywords)[:10])
        item_kw_df = pd.DataFrame({'keyword':list(keywords)})
        
        # user docs
        user_docs = df.groupby('user_id')['text_clean'].apply(lambda x: ' '.join(x))
        keyword_extractor = keyword_extractor_class(user_docs, n_gram_range=(1,1))
        keyword_extractor.extract_keywords(top_n=5)
        keywords = keyword_extractor.get_keywords(duplicate_limit=0.1, num_keywords=500)
        user_kw_df = pd.DataFrame({'keyword':list(keywords)})

        item_kw_df.to_csv(f'data/{data_name}/{keyword_extraction_method}_item_keywords.csv') 
        user_kw_df.to_csv(f'data/{data_name}/{keyword_extraction_method}_user_keywords.csv')

        nid_arr_dict = convert_doc_array(item_docs, user_docs, item_kw_df, user_kw_df)
        cooc_matrix = get_keyword_co_occurrence_matrix(nid_arr_dict)

        with open(f'data/{data_name}/{keyword_extraction_method}_cooc_matrix.npy', 'wb') as f:
            np.save(f, cooc_matrix)
