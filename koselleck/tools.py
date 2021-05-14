from .imports import *


def periodize_sattelzeit(y):
    if 1700<=y<1770: return '1700-1770'
    if 1770<=y<1830: return '1770-1830'
    if 1830<=y<1900: return '1830-1900'
    



def get_keywords():
    with open(PATH_FIELD_WILLIAMS_SRC) as f:
        return f.read().strip().lower().split()

def get_fields():
    with open('/home/ryan/github/0AbsLitHist/data/fields/data.fields.json') as f:
        fieldd={**json.load(f), **get_origfields()}
        return fieldd


def get_valid_words(only_pos={'nn1'},max_rank=25000,force=False,lim=None):
    if not force and os.path.exists(WORD_FN):
        with open(WORD_FN) as f:
            return f.read().strip().split('\n')
    C=get_corpus()
    mfwdf=C.mfw_df(only_pos=only_pos).query(f'ranks_avg<={max_rank}')
    words = set(mfwdf.index[:lim])
    with open(WORD_FN,'w') as of:
        of.write('\n'.join(words))
    return words


# def get_dfpiv_abs(words,wordvec_fn='wordvecs.Abs-Conc.txt',min_periods=3, pivot=True, z=False, min_year=1700,max_year=1900):
#     pathdf=get_model_paths_df(PATH_MODELS_OLD, wordvec_fn).sort_values('period_start')
#     o=[]
#     for dec,path in tqdm(zip(pathdf.period_start, pathdf.path), total=len(pathdf)):
#         dec=int(dec)
#         if dec<min_year or dec>=max_year: continue
#         ddf=pd.read_csv(path,sep='\t')
#         ddf['csim']=1-ddf['cdist']
        
#         ddf['csim_z']=(ddf['csim']-ddf['csim'].mean())/ddf['csim'].std()
#         ddf=ddf[ddf.word.isin(set(words))]
#         ddf['year']=int(dec)
#         o.append(ddf)
#     odf=pd.concat(o)
#     odf=odf.reset_index().groupby(['word','year']).mean().reset_index()
#     pivdf=odf.pivot('word','year','csim_z' if z else 'csim')
#     return pivdf


def get_dfpiv(key, fn=FN_VECTOR_SCORES_RUNS, ymin=1700, ymax=1900):
    df=pd.read_csv(fn)
    pwdf=df.groupby(['period','word']).mean().reset_index()
    pwdf['period']=[int(p[:4]) for p in pwdf.period]
    pwdf=pwdf.query(f'{ymin}<=period<{ymax}')
    pwdfpiv=pwdf.pivot('word','period',key)
    return pwdfpiv
    

def get_dfpiv_abs(key=FIELD_ABS_KEY,**y):
    return get_dfpiv(key)
    




DFPKG=None
def get_df_package():
    global DFPKG
    if DFPKG is not None: return DFPKG
        # load decade level data
    dfpiv_abs,dfpiv_freq = get_decade_level_data()
    # load diff data by run
    dfruns=pd.read_csv(FN_CHANGE_RUNS).set_index('word')
    # Load diffdata avg
    dfchange=get_dfchange()
    DFPKG=(dfchange,dfruns,dfpiv_abs,dfpiv_freq)
    return DFPKG


def get_dfpiv_freq(words,yearbin=10,year_min=1700,year_max=1900,z=False):
    C=get_corpus()
    dtm=C.dtm(words=words)
    dtm=dtm[[c for c in dtm.columns if not c in {'amp'}]]
    mdtm=C.meta[['year']].join(dtm,rsuffix='_w')
    mdtm=mdtm.query(f'{year_min}<=year<{year_max}')
    mdtm['year']=mdtm.year//yearbin*yearbin
    mdtmy=mdtm.groupby('year').sum()
    mdtmys=mdtmy.apply(lambda x: x/x.sum(),1)
    mdtmysT=mdtmys.T
    if z: mdtmysT=to_z(mdtmysT)
    return mdtmysT

def get_decade_level_data(words=None,z=True):
    if os.path.exists(FN_DATA_CACHE_DEC):
        with open(FN_DATA_CACHE_DEC,'rb') as f:
            return pickle.load(f)
    
    # load decade level data
    if words is None: words=get_valid_words()
    dfpiv_abs=get_dfpiv_abs(words=words,z=z).dropna()
    words=set(dfpiv_abs.index) 
    dfpiv_freq=get_dfpiv_freq(words,z=z).dropna()
    o=dfpiv_abs,dfpiv_freq
    with open(FN_DATA_CACHE_DEC,'wb') as of:
        pickle.dump(o,of)
    return o

def get_pathdf_models():
    pathdf=get_model_paths_df(PATH_MODELS_BPO, 'model.bin').sort_values(['period_start','run'])
    # pathdf['period']=pathdf.period_start.apply(periodize_sattelzeit)
    pathdf['period']=[f'{x}-{y}' for x,y in zip(pathdf.period_start, pathdf.period_end)]
    pathdf['period_len']=pathdf.period_end - pathdf.period_start
    return pathdf[~pathdf.period.isnull()]
#     return pathdf#.query('1700<=period_start<1900')


def classify_abstractness(row,perc_threshold=70,dist_threshold=0.5):
    # if row.score_diff_p_abstractness<=0.05 and row.dist_abstractness>=dist_threshold:
    if row.score_diff_p_abstractness<=0.05 and row.perc_abstractness>=perc_threshold:
        return '+Abstract' if row.score2_abstractness>row.score1_abstractness else '+Concrete'
    return 'Abs~Conc'
def classify_change(row,perc_threshold=70,dist_threshold=0.5):
    if row.is_clean_noiseaware:
        if row.perc_local>=perc_threshold:
        # if row.dist_local>=dist_threshold:
            return '+Changed'
    else:
        return '~Noisy'
    return '-Changed'


def get_dfchange(words=None):
    # Load collective difference data
    dfchange=pd.read_csv(FN_CHANGE_RUNS_AVG).set_index('word').sort_values('rank')
    dfchange['class_abs']=dfchange.apply(classify_abstractness,1)
    dfchange['class_change']=dfchange.apply(classify_change,1)
    dfchange['class_signif']=[(x!='Abs~Conc' or y=='+Changed') for x,y in zip(dfchange.class_abs, dfchange.class_change)]
    dfchange['class']=[f'{x} {y}' for x,y in zip(dfchange.class_abs, dfchange.class_change)]
    
    # filter?
    if words: dfchange=dfchange.loc[[w for w in words if w in set(dfchange.index)]]
    return dfchange.sort_values('rank')
    #.query('class_abs!="Abs~Conc" | class_change=="+Changed"')
    # return dfchange






def show_change_table(dfchange,gby=['class_abs','class_change']):
    pd.options.display.max_colwidth=100
    dfchange_f=dfchange.query('class_signif==True')
    for gname,gdf in dfchange_f.groupby(gby[0]):
        printm('## '+gname)
        gdfx=pd.DataFrame()
        for gname2,gdf2 in gdf.groupby(gby[1]):
    #         printm('### '+gname2)
            o=', '.join(gdf2.sort_values('rank_local').index)
    #         printm()
            gdfx[gname2]=[o]
        printm(gdfx.to_markdown())
    #     break





C=None
def get_corpus():
    global C
    if C is None:
        C=lltk.load(CNAME)
    return C

def load_model_row(row,**y):
    return load_model(row.path, row.path_vocab,**y)



#### Misc
def to_z(pivdf,axis=0):
    pivdf=pivdf.T if not axis else pivdf
    pivdfz=pd.DataFrame(index=pivdf.index, columns=pivdf.columns)
    for c in tqdm(pivdf.columns): pivdfz[c]=(pivdf[c] - pivdf[c].mean()) / pivdf[c].std()
    return pivdfz.T if not axis else pivdfz



C=get_corpus()
logger.remove()
logger.add(sys.stderr, format="{message}", filter='koselleck', level="INFO")
# logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
def log(*x,**y): logger.info(*x,**y)