from koselleck.imports import *
YEARBIN=5

MODEL_CACHE2={}

def load_model(path_model,path_vocab=None,min_count=None,cache_bin=True,cache=True):
    global MODEL_CACHE2
    
    if cache and path_model in MODEL_CACHE2: return MODEL_CACHE2[path_model]
    print('>> loading',path_model)
    path,model=do_load_model(path_model,path_vocab=path_vocab,min_count=min_count,cache_bin=cache_bin)
    return model
    

def get_vec_qstr(row_or_str,run=1):
    if type(run)==int: run=str(run).zfill(2)
    # get key
    if type(row_or_str) in {dict,pd.Series}:
        row=row_or_str
#         if 'qstr' in row:
#             qstr=row['qstr']
#         else:
        qstr=f'{row["period"]}_{row["run"]}'
    elif type(row_or_str)==str:
        row_or_str=row_or_str.split('(')[-1].split(')')[0]
        period=row_or_str.split('_')[0]
        run=run if not '_' in row_or_str else str(int(row_or_str.split('_')[1])).zfill(2)
        qstr=f'{period}_{run}'
    
    return f'vecs({qstr})'
    
    
# def get_vecs(row_or_str,run=1):
def vecs(period,run=1,words=[]):
    qstr=get_vec_qstr(period,run)
    res=dbget(qstr)
    if res is not None:
        if words: res=res.loc[[i for i in res.index if i in set(words)]]
    return res
    
    
def do_load_model(path_model,path_vocab=None,min_count=None,cache_bin=True):
#     print('>> loading',path_model)
    path_model_bin=path_model.split('.txt')[0]+'.bin' if not path_model.endswith('.bin') else path_model
    if os.path.exists(path_model_bin):
        model=gensim.models.KeyedVectors.load(path_model_bin,mmap='r')
    elif os.path.exists(path_model):
        if not path_vocab: path_vocab=os.path.join(os.path.dirname(path_model,'vocab.txt'))
        if os.path.exists(path_vocab):
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model,path_vocab)
            if min_count: filter_model(model,min_count=min_count)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model)
        if cache_bin:
            model.save(path_model_bin)
    else:
        print('!!??',path_model)
        stop
        return None,None
#     print(path_model, len(model.wv.key_to_index))
    return (path_model,model)
    
    
FN_DEFAULT_MODEL_PATHS=os.path.join(PATH_DATA,'data.model.paths.default.pkl')
def get_default_models():
    if os.path.exists(FN_DEFAULT_MODEL_PATHS):
        odf=read_df(FN_DEFAULT_MODEL_PATHS)
    else:
        odf=get_pathdf_models(period_len=5)
        odf.to_pickle(FN_DEFAULT_MODEL_PATHS)
    return odf.query('1720<=period_start<1960 & run<="run_10"')
    

def get_default_periods():
    return sorted(list(set(get_default_models().period)))

    
    
    
PRELOADED=None
def preload_models(paths=None,num_proc=1,all_model_cache=True):
    global MODEL_CACHE2,PRELOADED

    if PRELOADED is None:
        if not paths: paths=get_default_models().path
        
        objs=[p for p in paths if not p in MODEL_CACHE2]
        if len(objs):
            if all_model_cache and os.path.exists(FN_ALL_MODEL_CACHE):
                now=time.time()
                print('Reading from',FN_ALL_MODEL_CACHE)
                with open(FN_ALL_MODEL_CACHE,'rb') as f:
                    for k,v in pickle.load(f).items():
                        MODEL_CACHE2[k]=v
                print(f'Finished in {round(time.time()-now,1)} seconds')
            else:
                for p,m in pmap_iter(do_load_model, objs, num_proc=num_proc, desc='Preloading models'):
                    MODEL_CACHE2[p]=m
                if all_model_cache:
                    with open(FN_ALL_MODEL_CACHE,'wb') as of:
                        now=time.time()
                        print('Saving to',FN_ALL_MODEL_CACHE)
                        pickle.dump(MODEL_CACHE2,of)
                        print(f'Finished in {round(time.time()-now,1)} seconds')
        
        PRELOADED=True

        #return dict((path,MODEL_CACHE2.get(path)) for path in paths)
        
        
def get_path_model(corpus,period,run):
    return os.path.join(PATH_MODELS,corpus,period,f'run_{run:02}','model.bin')


# def save_vecs_in_db0(words=None,num_proc=1):
#     vl=get_veclib('vecs') 
#     dfmodels=get_default_models()
#     objs=list(zip(dfmodels.path,dfmodels.period,dfmodels.run))
#     for i,odx in enumerate(pmap_iter(
#         do_save_vecs_in_db,
#         objs,
#         num_proc=num_proc
#     )):
#         for k,v in odx.items(): vl[k]=v


def save_vecs_in_db(words=None,num_proc=1):
    with get_veclib('vecs',autocommit=False) as vl: 
        dfmodels=get_default_models()
        objs=list(zip(dfmodels.path,dfmodels.period,dfmodels.run))
        for i,odx in enumerate(pmap_iter(
            do_save_vecs_in_db,
            objs,
            num_proc=num_proc
        )):
            for k,v in odx.items(): vl[k]=v
            if i and not i%100: vl.commit()
        vl.commit()
            
def do_save_vecs_in_db(obj):
    vd={}
    path,prd,run = obj
    rnum=run.split('_')[-1]
    _,m=do_load_model(path)
    wl=[m.wv.index_to_key[i] for i in range(len(m.wv.vectors))]
    odf=pd.DataFrame(m.wv.vectors, index=wl)
    vd[f'vecs({prd}_{rnum})']=odf
    return vd