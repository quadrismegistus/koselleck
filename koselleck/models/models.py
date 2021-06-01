

def get_pathdf_models(period_len=5):
    pathdf=get_model_paths_df(PATH_MODELS_BPO, 'model.bin').sort_values(['period_start','run'])
    pathdf['period']=[f'{x}-{y}' for x,y in zip(pathdf.period_start, pathdf.period_end)]
    pathdf['period_len']=pathdf.period_end - pathdf.period_start
    if period_len: pathdf=pathdf[pathdf.period_len==period_len]
    return pathdf[~pathdf.period.isnull()]
