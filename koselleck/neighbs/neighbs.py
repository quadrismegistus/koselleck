from koselleck.imports import *


def to_nbr(word,period,max_rank=100,force=False,cache_only=False,num_proc=1,progress=True,cache_cdist=False,cache=True):
    qstr=f'{word}_{period}'
    odf=pd.DataFrame()
#     vl=get_veclib('nbr')
    with get_veclib('nbr',autocommit=True) as vl:
        if cache and qstr in vl:
            odf=vl.get(qstr)
        else:
            gby=['word','neighbor','period']
            dfcdist = cdist(word,period,num_proc=num_proc,progress=progress,cache=cache_cdist)
            dfprd=dfcdist.groupby('run').head(max_rank)
            dfprdg=dfprd.groupby(gby)
            dfprd=dfprd.reset_index().set_index(gby)
            dfprd['count']=dfprdg.size()
            dfprd['score']=[(c - (cd/10)) for c,cd in zip(dfprd['count'], dfprd['cdist'])]
            odf=dfprd.groupby(gby).mean()
            odf['rank']=odf['score'].rank(ascending=False,method='min').apply(int)
            odf=odf.drop('score',1).sort_values('rank')
            if max_rank: odf=odf[odf['rank']<=max_rank]
            if cache: vl[qstr]=odf
    return odf if not cache_only else pd.DataFrame()


def nbr_(argd):
    try:
        return to_nbr(**argd)
    except Exception as e:
        pass
#         print('!!',e,'!!')
#         pprint(argd)
    return pd.DataFrame()

def nbr(word_or_words,period_or_periods=None,prefix='nbr',neighbors=None,
        max_rank=100,force=False,cache_only=False,num_proc=4,cache_cdist=False):
    # preproc input
    words=tokenize_fast(word_or_words) if type(word_or_words)==str else list(word_or_words)
    if period_or_periods is None:
        periods=get_default_periods()
    elif type(period_or_periods)==str:
        periods=tokenize_fast(period_or_periods)
    else:
        periods=list(period_or_periods)
    # get objs
    objs = [
        dict(word=word,period=period,max_rank=max_rank,force=force,
             cache_only=cache_only,num_proc=1,progress=False,cache_cdist=cache_cdist)
        for word in words
        for period in periods
    ]
    # map
    return pd.concat(pmap(
        nbr_,
        objs,
        num_proc=num_proc,
        desc='Computing neighborhoods across word-periods'
    ))
    
    