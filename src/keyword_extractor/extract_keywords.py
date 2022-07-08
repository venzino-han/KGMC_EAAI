
import pandas as pd
import numpy as np
import nltk
from collections import defaultdict

from keyword_extractor import KeyBertExtractor




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


if __name__=='__main__':
    # load docs 
    data_name='game'
    df = pd.read_csv(f'data/{data_name}/{data_name}_fin.csv')
    df = df.dropna()

    # item docs
    item_docs = df.groupby('item_id')['text_clean'].apply(lambda x: ' '.join(x)).tolist()
    keybert_extractor = KeyBertExtractor(item_docs)
    keybert_extractor.extract_keywords(top_n=5)
    keywords = keybert_extractor.get_keywords(duplicate_limit=0.1, num_keywords=500)
    kw_df = pd.DataFrame({'keyword':list(keywords)})
    kw_df.to_csv(f'data/{data_name}/keybert_item_keywords.csv')
    
    # user docs
    user_docs = df.groupby('user_id')['text_clean'].apply(lambda x: ' '.join(x)).tolist()
    keybert_extractor = KeyBertExtractor(user_docs)
    keybert_extractor.extract_keywords(top_n=5)
    keywords = keybert_extractor.get_keywords(duplicate_limit=0.1, num_keywords=500)
    kw_df = pd.DataFrame({'keyword':list(keywords)})
    kw_df.to_csv(f'data/{data_name}/keybert_user_keywords.csv')