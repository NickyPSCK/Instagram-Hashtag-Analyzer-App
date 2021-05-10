# association_analyzer.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules 


# Reference: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

def count_object(dataset:list, unique_in_basket=False):
    counter = dict()
    for data in dataset:
        
        if unique_in_basket:
            data = set(data)
        
        for each in data:
            if each in counter:
                counter[each] += 1
            else:
                counter[each] = 1
    return counter

def calculate_support(dataset:list):

    try:
        count_object_dict = count_object(dataset, unique_in_basket=True)
        count_object_unique_in_basket_dict = count_object(dataset, unique_in_basket=False)

        result_df = pd.DataFrame(count_object_dict.items(), columns=['Object', 'Number of Object'])
        result_df['Number of Basket'] = result_df['Object'].map(count_object_unique_in_basket_dict)

        result_df['Support: Object'] = result_df['Number of Object']/result_df['Number of Object'].sum()
        result_df['Support: Basket'] = result_df['Number of Basket']/len(dataset)

        result_df.sort_values("Number of Basket", ascending=False)

    except Exception as inst:
        print('Warning')
        print(inst)         

        col_name = ['Object', 'Number of Object', 'Number of Basket', 'Support: Object', 'Support: Basket']
        result_df = pd.DataFrame(columns=col_name)

    return result_df

def calculate_association(dataset:list, 
                frequent_itemsets_algorithm:str='apriori',
                min_support:float = 0.3,
                association_metric:str = 'confidence',
                association_min_threshold:float = 1):
    try:
        te = TransactionEncoder()
        encoded_array = te.fit(dataset).transform(dataset)
        encoded_df = pd.DataFrame(encoded_array, columns=te.columns_)

        if frequent_itemsets_algorithm == 'apriori':
            frequent_itemsets_df = apriori(encoded_df, min_support=min_support, use_colnames=True)
        elif frequent_itemsets_algorithm == 'fpgrowth':
            frequent_itemsets_df = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
        elif frequent_itemsets_algorithm == 'fpmax':
            frequent_itemsets_df = fpmax(encoded_df, min_support=min_support, use_colnames=True)

        association_rules_df = association_rules(frequent_itemsets_df, metric=association_metric, min_threshold=association_min_threshold)
        
        # Formatting
        association_rules_df['antecedents'] = association_rules_df['antecedents'].apply(lambda x:[i for i in x])
        association_rules_df['consequents'] = association_rules_df['consequents'].apply(lambda x:[i for i in x])

        association_rules_df.sort_values(by='support', ascending=False, inplace = True)
        association_rules_df.reset_index(drop=True, inplace=True)

    
    except Exception as inst:
        print('Warning')
        print(inst)  

        col_name_frequent_itemsets_df = ['support', 'itemsets']
        frequent_itemsets_df = pd.DataFrame(columns=col_name_frequent_itemsets_df)

        col_name_association_rules_df = ['antecedents', 'consequents', 'antecedent support', 'consequent support', 
                                        'support', 'confidence', 'lift', 'leverage', 'conviction']

        association_rules_df = pd.DataFrame(columns=col_name_association_rules_df)
        
    return frequent_itemsets_df, association_rules_df
    
def check_objects(baskets, 
                    objects:list=None,
                    count:bool=False):

    if objects is None:
        objects = list()
    
    result_df = pd.DataFrame()
    result_df['basket'] = baskets

    if count:
        for obj in objects:
            result_df[obj] = result_df['basket'].apply(lambda basket: basket.count(obj))
    else:
        for obj in objects:
            result_df[obj] = result_df['basket'].apply(lambda basket: obj in basket)

    result_df = result_df.drop(['basket'], axis=1)

    return result_df

def summary_check_objects(check_df):

    no_of_basket = len(check_df)
    summary_df = check_df.sum().to_frame(name='Number of Basket')
    summary_df['Percent of Busket'] = (summary_df['Number of Basket']/no_of_basket) * 100

    return summary_df