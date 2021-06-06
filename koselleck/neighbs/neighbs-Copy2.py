# from koselleck.imports import *
# NBR='nbr'


# def save_neighb_to_veclib(odf,progress=True):
#     vl=get_veclib(prefix=NBR)
#     for word,wdf in tqdm(odf.groupby('word'),desc='Saving neighbors to veclib',disable=not progress):
#         vl[f'{NBR}({word})']=wdf.to_dict('records')


# def read_neighb_from_veclib(words,progress=True):
#     vl=get_veclib(prefix=NBR)
#     o=[]
#     for word in tqdm(words,desc='Reading neighbors from veclib',disable=not progress):
#         res=vl.get(f'{NBR}({word})')
#         if res is not None:
#             o+=res
#     return pd.DataFrame(o)

# def gen_neighbors(words=None, dfmodels=None, k=25, k_min=10, lim=None, num_proc=1, force=False, num_runs=10):
# #     if not force and os.path.exists(fnfn): return read_df(fnfn)
#     if not dfmodels: dfmodels = get_default_models()
#     if not words: words = get_valid_words()
#     if type(words)==str: words=tokenize_fast(words)

#     odf = odf_done = read_neighb_from_veclib(words)
#     words_todo = words if not len(odf_done) else list(set(words)-set(odf_done.word))

#     if len(words_todo):
# #         preload_models()

#         odf_new=pmap_groups(
#             do_gen_neighbs,
#             dfmodels.iloc[:lim].groupby(['corpus','period']),
#             num_proc=4,
#             desc='Gathering all neighborhoods',
#             use_cache=False,
#             kwargs=dict(k=k,words=words_todo)
#         ).reset_index()

#         save_neighb_to_veclib(odf_new)

#         odf=odf_done.append(odf_new)
#     #odf.to_pickle(FN_ALL_NEIGHBS)
#     return odf

# def _do_gen_neighbs(obj):
#     model_path,words,k = obj
#     m=load_model(model_path)
#     mwords=set(words)&set(m.wv.key_to_index.keys())
#     mdf=pd.DataFrame([
#         {'word':w, 'neighbor':w2, 'rank_avg':i+1, 'csim':c}
#         for w in mwords
#         for i,(w2,c) in enumerate(m.wv.most_similar(w,topn=k))
#     ])
#     return mdf

# def do_gen_neighbs(dfpath,words=None,k=25,progress=False,min_count=1):
#     model_path=dfpath.iloc[0].path
#     if not words: words=get_valid_words()
#     o=[]
# #     iter1=dfpath.path if not progress else tqdm(dfpath.path,desc='Iterating models',position=0)
#     objs=[(mpath,words,k) for mpath in dfpath.path]
#     o = pmap(_do_gen_neighbs, objs, num_proc=1, progress=progress)
#     if not len(o): return
#     odf=pd.concat(o)
#     gby=['word','neighbor']
#     odfg=odf.groupby(gby)
#     odf=odf.set_index(gby)
#     odf['count']=odfg.size()
#     odf=odf.query(f'count>={min_count}')#.set_index(['word'])
#     odf['score']=[c - (1/100) + (cs/1000)
#                   for c,r,cs in zip(odf['count'], odf.rank_avg, odf.csim)]
#     odf=odf.groupby(gby).mean().reset_index()
#     odf['rank']=odf.groupby('word')['score'].rank(ascending=False,method='min').apply(int)
#     odf=odf.sort_values(['word','rank'])#.drop('score',1)
#     return odf

# DF_ALLNEIGHB=None
# def get_all_neighbors(
#         fnfn=FN_ALL_NEIGHBS,
#         k=25,
#         k_min=10,
#         lim=None,
#         num_proc=1,
#         force=False,
#         num_runs=10,
#         min_count=1,
#         min_neighbs=1):
#     global DF_ALLNEIGHB
#     if DF_ALLNEIGHB is not None: return DF_ALLNEIGHB

#     fnfn_cache=fnfn.replace('.pkl','.cache.pkl')
#     if os.path.exists(fnfn_cache):
#         DF_ALLNEIGHB=read_df(fnfn_cache)
#         return DF_ALLNEIGHB

#     if not force and os.path.exists(fnfn):
#         print('Loading data')
#         odf=read_df(fnfn)
#     else:
#         odf=gen_all_neighbors(
#             fnfn=fnfn,
#             k=k,
#             k_min=k_min,
#             lim=lim,
#             num_proc=num_proc,
#             force=force,
#             num_runs=num_runs
#         )
#     odf=odf.drop('corpus',1).set_index(['word','period']).sort_index()
#     print('Filtering')
#     s=odf.query(f'count>={min_count}').groupby(['word','period']).neighbor.nunique()
#     print('Filtering, pt2')
#     odf=odf[s>=min_neighbs]
#     print('Postprocessing')
# #     odf=pd.concat(
# #         grp.rename({'rank':'rank_avg'}).assign(rank=[i+1 for i in range(len(grp))])
# #         for _,grp in odf.groupby(['word','period'])
# #     )
#     DF_ALLNEIGHB=odf
#     odf.to_pickle(fnfn_cache)
#     return odf










# def do_combine_neighbs(dfgrp,k=25,min_count=1):
#     dfgrp=dfgrp.reset_index()
#     dfgrp['count_str']=dfgrp['count']
#     dfgrp2=dfgrp.drop_duplicates('count')
#     firstwords=set(dfgrp2.neighbor)
#     dfgrp['count_str']=[f' ({int(c)})' if w in firstwords else ''
#                        for c,w in zip(dfgrp['count'], dfgrp['neighbor'])]    
#     return pd.DataFrame([{
#         'neighborhood':', '.join([
#             f'{n}{c}'
#             for n,c,r in zip(dfgrp.neighbor, dfgrp["count_str"], dfgrp['rank'])
#         ]),#[:k]),
#         'neighborhood_size':len(dfgrp)
#     }])

# FN_ALL_NEIGHBS_STR=FN_ALL_NEIGHBS.replace('.pkl','.strsummary.pkl')

# def get_all_neighbors_strsummary(dfneighbs=None,ofnfn=FN_ALL_NEIGHBS_STR,lim=None,k=25,num_proc=1,force=False,**y):
#     if not force and os.path.exists(ofnfn): return read_df(ofnfn)
#     if dfneighbs is None: dfneighbs=get_all_neighbors()
#     odf=pmap_groups(
#         do_combine_neighbs,
#         dfneighbs.iloc[:lim].groupby(['word','period']),
#         kwargs=dict(k=k),
#         num_proc=num_proc,
#         **y
#     )
#     odf.to_pickle(FN_ALL_NEIGHBS_STR)
#     return odf



# def get_all_neighbors_simple(fnfn_cache=FN_ALL_NEIGHBS_SIMPLE):
#     global NEIGHB_SIMPLE_D
#     if NEIGHB_SIMPLE_D is None:
#         if os.path.exists(fnfn_cache):
#             with open(fnfn_cache,'rb') as f:
#                 NEIGHB_SIMPLE_D=pickle.load(f)
#         else:
#             odx={}
#             df=get_all_neighbors_strsummary()
#             for i in tqdm(df.index): 
#                 odx[i]=[
#                     w.split()[0]
#                     for w in df.loc[i].neighborhood.split(', ')
#                 ]
#             with open(fnfn_cache,'wb') as of: pickle.dump(odx,of)
#             NEIGHB_SIMPLE_D=odx
#     return NEIGHB_SIMPLE_D
