from koselleck.imports import *

FORCE=False
YMIN=1720
YMAX=1960
GBY_LOCAL_O=['corpus1', 'corpus2','period1', 'period2', 'word1', 'word2','qstr']



def measure_dist_local(m1,m2,w1,w2,p1,p2,k=25,mwords=None):
    odx={}
    if not mwords: mwords=set(m1.wv.key_to_index.keys())&set(m2.wv.key_to_index.keys())
    #if w1 not in mwords or w2 not in mwords: return odx
    
    dfn=get_all_neighbors_simple()
    try:
        neighb1=[w for w in dfn[(w1,p1)] if w in mwords][:k]
        neighb2=[w for w in dfn[(w2,p2)] if w in mwords][:k]
        metaneighb=list(set(neighb1)|set(neighb2))
        wvec1=m1.wv.vectors[m1.wv.key_to_index[w1]]
        wvec2=m2.wv.vectors[m2.wv.key_to_index[w2]]
        nvecs1=np.array([m1.wv.vectors[m1.wv.key_to_index[w]] for w in metaneighb])
        nvecs2=np.array([m2.wv.vectors[m2.wv.key_to_index[w]] for w in metaneighb])
        vector1=1-fastdist.vector_to_matrix_distance(
            wvec1.astype(float), nvecs1.astype(float), fastdist.cosine, 'cosine'
        )
        vector2=1-fastdist.vector_to_matrix_distance(
            wvec2.astype(float), nvecs2.astype(float), fastdist.cosine, 'cosine'
        )
        csim=fastdist.cosine(vector1,vector2) # returns similarity not distane!!
        dist=1-csim
        odx={
            'dist_local':dist,
            'metaneighb_size':len(metaneighb),
            'neighb1_size':len(neighb1),
            'neighb2_size':len(neighb2),
        }
    except KeyError:
        pass
    return odx


def prep_dists_local_input(words,words2=[],corpus='bpo',corpus2=None,ymin=1720,ymax=1960,ybin=5,veclib=None,progress=False):
    if veclib is None: veclib=get_veclib()
    words1=words
    corpus1=corpus
    if type(words1)==str: words1=[words1]
    if not corpus2: corpus2=corpus1
    if not words2: words2=words1
    periods = [f'{y}-{y+ybin}' for y in range(ymin,ymax,ybin)]
    objs=[]
#     for period1,period2 in product(periods,periods):
    for period1,period2 in tqdm(list(product(periods,periods)),desc='Preparing input',disable=not progress):
        if period1>=period2: continue
        for word1,word2 in zip(words1,words2):
            qstr=f'lnm({word1}_{period1}_{corpus1},{word2}_{period2}_{corpus2})'
            odx=dict(
                period1=period1,
                period2=period2,
                corpus1=corpus1,
                corpus2=corpus2,
                word1=word1,
                word2=word2,
                qstr=qstr
            )
            objs.append(odx)
    dfinp=pd.DataFrame(objs)
    dfinp['done']=[(x in veclib) for x in tqdm(dfinp.qstr,desc='Checking if cached',disable=not progress)]
    return dfinp

def dists_local(words,words2=[],corpus='bpo',corpus2=None,ymin=1720,ymax=1960,ybin=5,num_runs=10,k=25,num_proc=1,progress=True):
    
    now=time.time()
    odf_done,odf_undone = pd.DataFrame(),pd.DataFrame()
    
    veclib=get_veclib()
    dfinp = prep_dists_local_input(words,words2,corpus,corpus2,ymin,ymax,ybin,veclib=veclib,progress=progress)
    idf_done,idf_undone = dfinp[dfinp.done==True].drop('done',1), dfinp[dfinp.done==False].drop('done',1)

    if len(idf_done):
#         print(f'Already measured {len(idf_done):,} local neighborhood distances')
        odf_done = idf_done.merge(
            pd.DataFrame({'qstr':qstr, **veclib[qstr]} for qstr in idf_done.qstr),
            on='qstr'
        )

    if len(idf_undone):
#         print(f'Measuring {len(idf_undone):,} local neighborhood distances')
        preload_models()
        odf_undone = pmap_groups(
            do_dists_local,
            idf_undone.groupby('period1'),
            num_proc=num_proc,
            kwargs=dict(k=k,num_runs=num_runs),
            desc='Measuring local neighborhood distances across periods',
            progress=progress
        )
        qstrdf=odf_undone.groupby('qstr').mean()
        for q,row in tqdm(qstrdf.iterrows(),desc='Adding to vector library',total=len(qstrdf)):
            veclib[q]=dict(row)
    else:
        odf_undone=pd.DataFrame()



    odf=odf_done.append(odf_undone)
    if len(odf): odf=odf.groupby(GBY_LOCAL_O).mean()
#     print(f'Finished in {round(time.time()-now,1)} seconds')
    return odf

    
def do_dists_local(dfinp,num_runs=10,k=25,progress=False):
    o=[]
    for nr in tqdm(list(range(1,num_runs+1)),desc='Measuring distances across runs',disable=progress):
        for (period1,corpus1),prd1df in dfinp.groupby(['period1','corpus1']):
            m1=load_model(get_path_model(corpus1,period1,nr))
            mwords1=set(m1.wv.key_to_index.keys())
            for (period2,corpus2),prd2df in prd1df.groupby(['period2','corpus2']):
                m2=load_model(get_path_model(corpus2,period2,nr))
                mwords2=set(m2.wv.key_to_index.keys())
                mwords=mwords1&mwords2
                
                for (word1,word2,qstr),wwdf in prd2df.groupby(['word1','word2','qstr']):
                    odx=measure_dist_local(m1,m2,word1,word2,period1,period2,k=k,mwords=mwords)
                    o+=[{**dict(wwdf.iloc[0]), **odx}]
#                     o+=[odx]
    return pd.DataFrame(o)    



def get_historical_semantic_distance_matrix(
        words=None,
        df_dists=None,
        dist_key='dist_local',
        ymin=None,
        ymax=None,
        interpolate=False,
        normalize=False,
        num_proc=1,
        progress=True):
    
    if df_dists is None:
        if type(words)==str: words=tokenize_fast(words)
        df_dists=dists_local(
            words,
            num_proc=num_proc,
            progress=progress
        )
    odfi=df_dists.groupby(['period1','period2']).mean().reset_index()
    odfi=odfi.append(odfi.assign(period1=odfi.period2,period2=odfi.period1))
    odfi[f'{dist_key}_perc']=odfi[dist_key].rank(ascending=True) / len(odfi) * 100
    odfp=odfi.pivot('period1','period2',f'{dist_key}_perc')#.fillna(0)
    if interpolate:
        for idx in odfp.index:
            odfp.loc[idx] = odfp.loc[idx].interpolate(limit_direction='both')
    return odfp





def plot_distmat(distdf,xcol='period1',ycol='period2',value_name='dist_local_perc',use_color=False,xlim=None,ylim=None,title='Distance matrix',ofn=None,invert=False,**y):
    distdfm=distdf.reset_index().melt(id_vars=[xcol],value_name=value_name).dropna()
    
    fig=start_fig(
        distdfm,
        x=f'factor({xcol})',
        y=f'factor({ycol})',
        fill=value_name,
        **y
    )
    fig+=p9.geom_tile()
    if not use_color:
        if not invert:
            fig+=p9.scale_fill_gradient(high='#111111',low='#FFFFFF')   
        else:
            fig+=p9.scale_fill_gradient(low='#111111',high='#FFFFFF')   
    else:
        fig+=p9.scale_fill_distiller(type='div',palette=5)
    fig+=p9.theme(
        axis_text_x=p9.element_text(angle=90)
    )
    fig+=p9.labs(
        x='Date of semantic model',
        y='Date of semantic model',
        fill='Semantic distance\n(LNM percentile)',
        title=title
    )
    if ofn:
        ofnfn=os.path.join(PATH_FIGS,ofn)
        fig.save(ofnfn)
        upfig(ofnfn)
        #fig.save(os.path.join('/home/ryan/Markdown/Drafts/TheGreatAbstraction/figures', ofn))
        
    return fig


def plot_historical_semantic_distance_matrix(words,save=False,dist_key='dist_local',interpolate=True,vnum='v34',**y):
    wstr=words.strip() if type(words)==str else '-'.join(words)
    wstr2=words.strip() if type(words)==str else ', '.join(words)
    return plot_distmat(
        get_historical_semantic_distance_matrix(
            words,
            dist_key=dist_key,
            interpolate=True,
            **y
        ),
        figure_size=(8,8),
        ofn=f'fig.distmat.{wstr}.{vnum}.png' if save else None,
        title=f'Historical-semantic distance matrix for ‘{wstr2}’',
        
    )









def get_default_periods():
    return sorted(list(set(get_default_models().period)))

def do_cdist(objd):
    res=cdist(**objd)
    if res is None: res=pd.DataFrame()
    return res

def cdist(word,period=None,run=None,prefix='cdist',words=None,max_num=None,num_runs=10,num_proc=1):
    vl=get_veclib(prefix)
#     if type(word)!=str:
#         objs=[
#             dict(word=w,period=period,run=run,prefix=prefix,words=words,max_num=max_num,num_runs=num_runs,num_proc=1)
#             for w in word
#         ]
#         return pd.concat(pmap(do_cdist, objs, num_proc=num_proc))

    if period is None:
        objs=[
            dict(word=word,period=period,run=run,prefix=prefix,words=words,max_num=max_num,num_runs=num_runs,num_proc=1)
            for period in get_default_periods()
        ]
        return pd.concat(pmap(do_cdist, objs, num_proc=num_proc))

    if run is None:
        objs=[
            dict(word=word,period=period,run=run+1,prefix=prefix,words=words,max_num=max_num,num_runs=num_runs,num_proc=1)
            for run in range(num_runs)
        ]
        return pd.concat(pmap(do_cdist, objs, num_proc=num_proc, progress=False))

    # otherwise

    # get?
    if type(run)==int: run=str(run).zfill(2)
    wqstr=f'{prefix}({word}_{period}_{run})'
    if wqstr in vl and type(vl[wqstr])==pd.DataFrame:
        wddf=vl[wqstr]
    else:
        # gen
        dfvecs=get_vecs(period=period, run=run, words=words)
        if dfvecs is None:
            print(wqstr,'!?')
            return pd.DataFrame()
        if not words: words=dfvecs.index
        words=set(words)
        if not word in words: return pd.DataFrame()    
        dfu=dfvecs.loc[word]
        if max_num and len(dfvecs)>max_num: dfvecs=dfvecs.iloc[:max_num]
        dfm=dfvecs.drop(word)
        res=fastdist.cosine_vector_to_matrix(
            dfu.values.astype(float),
            dfm.values.astype(float),
        )
        wdx=dict(
            (x,1-y)
            for x,y in zip(dfm.index, res)
        )
        wds=pd.Series(wdx).sort_values()
        wddf=pd.DataFrame(wds,columns=['cdist']).rename_axis('word')
        vl[wqstr]=wddf
    wddf=wddf.reset_index()
    wddf['period']=period
    wddf['run']=run
    wddf=wddf.set_index(['word'])#,'period','run'])
    return wddf
