from collections import defaultdict
import pandas as pd
import pickle as pkl
from itertools import combinations, product

def get_id_keyword_dict(df)->defaultdict:
    keyword_id_dict = defaultdict(set)
    for row in df.iterrows():
        id, kws = row[0], row[1].split()
        for kw in kws:
            keyword_id_dict[kw].add(id)
    return keyword_id_dict
    
def get_cooc_pair_df(keyword_id_dict)->pd.DataFrame:
    """
    return cooc edge as df format
    uid, vid, cooc_count
    """
    pair_dict = defaultdict(int)
    for kw, ids in keyword_id_dict.items():
        for u, v in combinations(ids, 2):
            if u > v:
                u,v=v,u
            pair_dict[(u,v)] += 1

    us, vs, cs = [], [], []
    for k, c in pair_dict.items():
        u, v = k 
        us.append(u)
        vs.append(v)
        cs.append(c)

    return pd.DataFrame({
        'u':us,
        'v':vs,
        'c':cs,
    })


def get_uv_cooc_pair_df(u_dict, v_dict)->pd.DataFrame:
    """
    return cooc edge between `user-item` as df format
    uid, vid, cooc_count
    """
    pair_dict = defaultdict(int)
    for kw, u_ids in u_dict.items():
        if kw not in v_dict:
            continue
        v_ids = v_dict.get(kw,set())
        for u, v in product(u_ids, v_ids):
            pair_dict[(u,v)] += 1

    us, vs, cs = [], [], []
    for k, c in pair_dict.items():
        u, v = k 
        us.append(u)
        vs.append(v)
        cs.append(c)

    return pd.DataFrame({
        'u':us,
        'v':vs,
        'c':cs,
    })


if __name__=='__main__':
    for keyword_extraction_method in ['text_rank','topic_rank']:
        for data_name in [
                        'yelp',
                        'grocery',
                        'epinions',
                        'games', 
                        'music', 
                        'office', 
                        'sports',
                        ]:
            
            item_df = pd.read_csv(f'data/{data_name}/item_{keyword_extraction_method}_keywords.csv')
            user_df = pd.read_csv(f'data/{data_name}/user_{keyword_extraction_method}_keywords.csv')

            cooc_dict = get_id_keyword_dict(item_df)
            cooc_df = get_cooc_pair_df(cooc_dict)
            cooc_df.to_csv(f'data/{data_name}/item_{keyword_extraction_method}_cooc.csv')

            cooc_dict = get_id_keyword_dict(user_df)
            cooc_df = get_cooc_pair_df(cooc_dict)
            cooc_df.to_csv(f'data/{data_name}/user_{keyword_extraction_method}_cooc.csv')

            cooc_dict = get_id_keyword_dict(user_df)
            cooc_df = get_uv_cooc_pair_df(cooc_dict)
            cooc_df.to_csv(f'data/{data_name}/user_item_{keyword_extraction_method}_cooc.csv')


            # keyword_item_dict = defaultdict(set)
            # keyword_user_dict = defaultdict(set)

            # for row in item_df.iterrows():
            #     iid, kws = row.item_id, row.keywords.split()
            #     for kw in kws:
            #         keyword_item_dict[kw].add(iid)

            # for row in user_df.iterrows():
            #     uid, kws = row.user_id, row.keywords.split()
            #     for kw in kws:
            #         keyword_user_dict[kw].add(uid)

            

            # with open(f'data/{data_name}/item_{keyword_extraction_method}_kw_dict.pkl', 'wb') as f:
            #     pkl.dump(keyword_item_dict, f)
            # with open(f'data/{data_name}/user_{keyword_extraction_method}_kw_dict.pkl', 'wb') as f:
            #     pkl.dump(keyword_user_dict, f)            
