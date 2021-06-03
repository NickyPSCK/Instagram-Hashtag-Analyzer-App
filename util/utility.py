
# util/utility.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------


def round_df(df:object, decimals:int=None):
    # https://datatofish.com/round-values-pandas-dataframe/
    dtype_dict = df.dtypes.iloc[1:].to_dict()
    for col_name in dtype_dict:
        if 'float' in str(dtype_dict[col_name]):
            if decimals is not None:
                df[col_name] = df[col_name].round(decimals=decimals)
    return df

