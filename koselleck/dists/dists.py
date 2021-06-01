from koselleck.imports import *

def get_all_localdists():
    global DF_LOCALDISTS
    if DF_LOCALDISTS is None: DF_LOCALDISTS=get_cross_model_dists().reset_index()
    #DF_LOCALDISTS=pd.read_pickle(FN_ALL_LOCALDISTS).groupby(['period1','period2','word']).mean()
    return DF_LOCALDISTS



"""
GENERATING WITHIN MODEL DISTANCES
"""

def do_gen_within_model_dists(pathdf,words=None,force=False):
    mpath=pathdf.iloc[0].path
    ofnfn=os.path.join(os.path.dirname(mpath), 'dists.pkl')
    res=pd.DataFrame([{'path_dists':ofnfn}])
    if not force and os.path.exists(ofnfn): return res
    
    if not os.path.exists(mpath): return pd.DataFrame()
    m=load_model(mpath)

    if not words: words=get_valid_words()
    words=list(set(words) & set(m.wv.key_to_index.keys()))
    
#     vecs=m.wv.get_normed_vectors()
    vecs=m.wv.vectors
    dfmat = pd.DataFrame([
        vecs[m.wv.key_to_index[w]]
        for w in words
    ],index=words)
    
    mat_cos = fastdist.cosine_pairwise_distance(dfmat.values,return_matrix=True)
    dfsims = pd.DataFrame(mat_cos, index=words, columns=words)
    dfdist=2-dfsims
    dfdist=round(dfdist,8)
    dfdist.to_pickle(ofnfn)
    return res


def gen_within_model_dists(
        dfmodels=None,
        words=None,
        num_proc=1,
        period_len=5,
        ymin=None,
        ymax=None,
        force=False):
    dfmodels = get_pathdf_models()
    dfmodels['path_dists']=[os.path.join(os.path.dirname(p),'dists.pkl') for p in dfmodels.path]
    dfmodels = dfmodels[dfmodels.period_len==period_len]
    if ymin: dfmodels=dfmodels[dfmodels.period_start>=ymin]
    if ymax: dfmodels=dfmodels[dfmodels.period_end<=ymax]
    if not force: dfmodels=dfmodels[dfmodels.path_dists.apply(lambda p: not os.path.exists(p))]
    if not len(dfmodels):
        print('All within-model distance matrices already saved')
        return
    print(f'Generating internal distances for models')
    if not words: words=get_valid_words()
    return pmap_groups(
        do_gen_within_model_dists,
        dfmodels.groupby(['corpus','period','run']),
        num_proc=num_proc,
        kwargs=dict(
            words=words
        ),
        desc='Generating internal distances for models'
    )




"""
ACROSS MODEL DISTANCES
"""

def get_cross_model_dists_paths(ofn='data.all_local_dists.paths.csv',force=False,period_len=5,ymin=1720,ymax=1960):
    ofnfn=os.path.join(PATH_DATA,ofn)
    if not force and os.path.exists(ofnfn): 
        odf=read_df(ofnfn)
    else:
        dfpaths=get_model_paths_df(PATH_MODELS_BPO, 'dists.pkl').query(
            f'(period_end-period_start)==5 & period_start>={ymin} & period_end<={ymax}'
        ).sort_values('period_start')
        dfpaths['period']=[f'{x}-{y}' for x,y in zip(dfpaths.period_start, dfpaths.period_end)]
#         display(dfpaths)
        o=[]
        for i1,row1 in tqdm(dfpaths.iterrows(), total=len(dfpaths)):
            for i2,row2 in dfpaths.iterrows():
                if row1.run!=row2.run: continue
                if i1>=i2: continue
                o+=[{
                    **dict((k+'1',v) for k,v in row1.items()),
                    **dict((k+'2',v) for k,v in row2.items())
                }]
        odf=pd.DataFrame(o)
        odf.to_csv(ofnfn,index=False)
    return odf.sort_values(['period_start1','period_start2','run1','run2'])


def gen_cross_model_dists(
        dfpaths_cmp=None,
        lim=None,
        num_proc=4,
        num_runs=1,
        ofnfn=FN_ALL_LOCALDISTS_V2,
        force=False,
        **y):
    if not force and os.path.exists(ofnfn):
        odf=read_df(ofnfn)
    else:
        if dfpaths_cmp is None: dfpaths_cmp=get_cross_model_dists_paths()
        dfpaths_cmp_f = dfpaths_cmp.query(f'run1<="run_{str(num_runs).zfill(2)}" & run2<="run_{str(num_runs).zfill(2)}"')
        odf=pmap_groups(
            do_gen_cross_model_dists,
            dfpaths_cmp_f.iloc[:lim].groupby(['corpus1','period1','run1']),
            num_proc=num_proc,
            desc='Calculating Local Neighborhood Distance Measure over periods',
            **y
        )
        odf.to_pickle(ofnfn)
    return odf

def do_gen_cross_model_dists(pathdf,progress=False,ks=[10,25,50],progress_words=False):
    row=pathdf.iloc[0]
    dfdist1_orig=read_df(row.path1)
    
    o=[]
    iterr=tqdm(pathdf.path2,position=0) if progress else pathdf.path2
#     display(pathdf.shape)
    for i2 in range(1,len(pathdf)):
        row2=pathdf.iloc[i2]
        dfdist2_orig=read_df(row2.path2)
        words = list(set(dfdist1_orig.columns) & set(dfdist2_orig.columns))
        dfdist1=dfdist1_orig[words].loc[words]
        dfdist2=dfdist2_orig[words].loc[words]

        iter2 = tqdm(words,position=0) if progress_words else words
        for w in iter2:
            neighb1_all=list(dfdist1[w].sort_values().index)
            neighb2_all=list(dfdist2[w].sort_values().index)
            
            for k in ks:
                neighb1=neighb1_all[:k+1]
                neighb2=neighb2_all[:k+1]
                metaneighb=list(set(neighb1)|set(neighb2))
                vector1=[dfdist1[w].loc[wx] for wx in metaneighb]
                vector2=[dfdist2[w].loc[wx] for wx in metaneighb]
                csim=fastdist.cosine(vector1,vector2) # returns similarity not distane!!
                dist=1-csim
                o+=[{
                    'corpus2':row2.corpus2,
                    'period2':row2.period2,
                    'run2':row2.run2,
                    'word':w,
                    'dist_local':dist,
                    'k':k
                }]
    return pd.DataFrame(o)
    
def get_cross_model_dists(fnfn_cache=FN_ALL_LOCALDISTS_V2_CACHE,cache=True,force=False,**y):
    if cache and not force and os.path.exists(fnfn_cache): return read_df(fnfn_cache)
    
    odf=read_df(FN_ALL_LOCALDISTS_V2)
    odf['k']=odf['k'].apply(int)
    odf_z = pd.concat(
        dfg.assign(
            dist_local_z = (dfg.dist_local - dfg.dist_local.mean()) / dfg.dist_local.std(),
            dist_local_perc = dfg.dist_local.rank() / len(dfg.dist_local) * 100,
            #dist_local_perc = dfg.dist_local.apply(lambda x: percentileofscore(dfg.dist_local, x))
        )
        for i,dfg in tqdm(odf.groupby('k'),desc='Normalizing scores by k-value')
#         for i,dfg in odf.groupby('k')
    ).reset_index()
    
    # average out runs
    odf_z_mean = odf_z.groupby(['corpus1','corpus2','period1','period2','word']).mean().drop('k',1).sort_values('dist_local')
    if cache: odf_z_mean.to_pickle(fnfn_cache)
    return odf_z_mean
    
    


    
    
    
    
    

"""
Distance matrix functions
"""



def get_historical_semantic_distance_matrix(
        words=None,
        dist_key='dist_local_perc',
        ymin=None,
        ymax=None,
        interpolate=False,
        normalize=False,
        ks={10,25,50}):
    df=get_all_localdists()
    if type(words)==str: words=tokenize_fast(words)
    if words: df=df[df.word.isin(words)]
    if ymin: df=df.query(f'period1>="{ymin}" & period2>="{ymin}"')
    if ymax: df=df.query(f'period1<"{ymax}" & period2<"{ymax}"')
#     if ks: df=df[df.k.isin(ks)]
    
    # fill out other half
    pdf=df.groupby(['period1','period2']).mean().reset_index()
    pdf2=pd.DataFrame(pdf).rename(columns=dict(period1='period2',period2='period1'))
    odf=pdf.append(pdf2)
    period_types=set(odf.period1) | set(odf.period2)
    pdf3=pd.DataFrame([{'period1':x, 'period2':x, dist_key:0} for x in period_types])
    odf=odf.append(pdf3)
    odf[dist_key]=odf[dist_key] / odf[dist_key].max()
    odf=odf.pivot('period1','period2',dist_key)
    if interpolate:
        for idx in odf.index:
            odf.loc[idx] = odf.loc[idx].interpolate(limit_direction='both')
    return odf








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


def plot_historical_semantic_distance_matrix(words,save=False,dist_key='dist_local_perc',interpolate=True,vnum='v34',**y):
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