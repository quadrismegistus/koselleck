from .imports import *


def periodize_sattelzeit(y):
    if 1700<=y<1770: return '1700-1770'
    if 1770<=y<1830: return '1770-1830'
    if 1830<=y<1900: return '1830-1900'

def periodize_sattelzeit_binary(y):
    if 1700<=y<1770: return '1700-1770'
    if 1830<=y<1900: return '1830-1900'
    
UPROOT='/Markdown/Drafts/TheGreatAbstraction/figures/'
def upfig(fnfn,uproot=UPROOT):
    ofnfn=os.path.join(uproot,os.path.basename(fnfn))
    os.system(f'dbu upload {fnfn} {ofnfn}')
    return os.system(f'dbu share {ofnfn}')

def get_keywords(url=URL_KEYWORDS,just_words=False):
    df=pd.read_csv(url).fillna('')
    df['word']=df.word.apply(lambda x: x.lower())
    # df=df[~df.word.isin({'?',''})]
    return df.set_index('word') if not just_words else set(df.word)

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

def get_words_ever_abs():
    dfc=get_dfchange()
    dfc12=get_dfchange_simple(dfc, 'mean1').append(get_dfchange_simple(dfc, 'mean2'))
    words_ever_abs=set(dfc12[dfc12['Abs-Conc.Median']>1].index)
    return words_ever_abs

def classify_vector_change(row,perc_threshold=75,z_threshold=1,z_threshold2=.5):
    name=row.vector
    ns=str(name).split('.')[0].split('-')
    n1,n2=ns[0],ns[-1]
    # res='~'#f'{n1}~{n2}'
    res='+' if row.mean_diff>0 else '-'
    if row.mean_diff_p<0.05 and row.perc>perc_threshold:
        if row.mean_diff>0:
            ## increases
            # decreases
            if row.mean2<-z_threshold and row.mean1<-z_threshold:
                # still over the line for conc
                res=f'{n2}-'
            elif row.mean2>z_threshold:
                # we're over the line for pos
                if row.mean1>z_threshold:
                    # already was though
                    res=f'{n1}+'                    
                elif row.mean1<-z_threshold:
                    # jumped all the way from neg to pos!
                    res=f'{n1}+++' 
                elif row.mean1<z_threshold2:
                    # jumped part of the way to become new
                    res=f'{n1}++'
                else:
                    res=f'{n1}+'
            elif row.mean1<-z_threshold:
                # really this is a loss of the prior
                res=f'{n2}--'
            else:
                res=f'{n1}+'
        else:
            # decreases
            if row.mean2>z_threshold and row.mean1>z_threshold:
                # still over the line for abs
                res=f'{n1}-'
            elif row.mean2<-z_threshold:
                # over the line for neg
                if row.mean1<-z_threshold:
                    # already was tho
                    res=f'{n2}+'
                elif row.mean1>z_threshold:
                    # jumped all the way from pos to neg
                    res=f'{n2}+++'
                elif row.mean1>-z_threshold2:
                    # jumped part of the way to become new
                    res=f'{n2}++'
                else:
                    res=f'{n2}+'
            elif row.mean1>z_threshold:
                # really this is a loss of the prior
                res=f'{n1}--'
            else:
                res=f'{n2}+'
    return res

DFCC=None
DFCC_CACHE_FN=os.path.join(PATH_DATA,'data.classed_changes.pkl')
def get_classed_changes(cache=True):
    global DFCC
    if DFCC is None:
        if cache and os.path.exists(DFCC_CACHE_FN):
            df=pd.read_pickle(DFCC_CACHE_FN)
        else:
            df=get_dfchange_from_decade_data(wide=False).reset_index()
            df['perc']=df.mean_diff_t_abs.apply(lambda x: percentileofscore(df.mean_diff_t_abs, x))
            dfchsz=get_dfchange_from_sattelzeit_models().reset_index()
            df['change']=df.progress_apply(classify_vector_change,axis=1)
            dfchsz['change']=dfchsz['class_change']
            df=df.append(dfchsz)
            startcols=['word','vector','change']
            df=df[startcols + [c for c in df.columns if c not in set(startcols)]]
            df['mean_diff_abs']=df['mean_diff'].apply(abs)
            df['change_rank']=[
                x.count('+') + x.count('-') if len(x)>1 else 0
                for x in df.change
            ]
            if cache: df.to_pickle(DFCC_CACHE_FN)
        DFCC=df
    return DFCC.sort_values(['change_rank','mean_diff_abs'],ascending=False)



def get_dfpiv(key, df=None, fn=FN_VECTOR_SCORES_RUNS, ymin=1700, ymax=1900, words=None, z=False, axis=1):
    if df is None: df=pd.read_pickle(fn)
    pwdf=df.groupby(['period','word']).mean().reset_index()
    pwdf['period']=[int(p[:4]) for p in pwdf.period]
    pwdf=pwdf.query(f'{ymin}<=period<{ymax}')
    pwdfpiv=pwdf.pivot('word','period',key)
    if words: pwdfpiv=pwdfpiv.reset_index()[pwdfpiv.reset_index().word.isin(set(words))].set_index('word')
    if z: pwdfpiv=to_z(pwdfpiv,axis=axis)
    return pwdfpiv
    

def get_dfpiv_abs(key=FIELD_ABS_KEY,**y):
    return get_dfpiv(key,**y)
    
def get_pathdf_models_bydecade(ymin=1700,ymax=1900):
    return get_pathdf_models().query(f'period_len==10 & {ymin}<=period_start<{ymax}')
def get_pathdf_models_byyear(ymin=1700,ymax=1900):
    return get_pathdf_models().query(f'period_len==2 & {ymin}<=period_start<{ymax}')




DFPKG=None
def get_df_package():
    global DFPKG
    if DFPKG is not None: return DFPKG
        # load decade level data
    dfpiv_abs,dfpiv_freq = get_decade_level_data()
    # load diff data by run
    dfruns=pd.read_csv(FN_CHANGE_RUNS).set_index('word')
    dfruns_dec=pd.read_pickle(FN_VECTOR_SCORES_RUNS)
    # Load diffdata avg
    dfchange=get_dfchange()
    DFPKG=(dfchange,dfruns,dfpiv_abs,dfpiv_freq,dfruns_dec,get_dfpiv_ambig())
    return DFPKG

def get_dfpiv_ambig(fn=FN_AMBIGUITY,z=True):
    ambdf=pd.read_csv(FN_AMBIGUITY)
    odf=ambdf.groupby('period').mean()
    odf.index=[int(p[:4]) for p in odf.index]
    odf=odf.T
    if z: odf=to_z(odf)
    return odf

def get_dfpiv_freq(words=None,ymin=1700,ymax=1900,z=False):
    df=pd.read_csv(FN_FREQ_DEC_MODELS).set_index('word')
    df.columns=[int(x[:4]) for x in df.columns]
    df=df[[y for y in df.columns if y>=ymin and y<ymax]]
    if words: df=df.reset_index()[df.reset_index().word.isin(set(words))].set_index('word')
    if z: df=to_z(df, axis=0)
    return df

def get_decade_level_data(words=None,z=True):
    if os.path.exists(FN_DATA_CACHE_DEC):
        with open(FN_DATA_CACHE_DEC,'rb') as f:
            return pickle.load(f)
    
    # load decade level data
    if words is None: words=get_valid_words()
    dfpiv_abs=get_dfpiv_abs(words=words,z=z)#.dropna()
    dfpiv_freq=get_dfpiv_freq(words=words,z=z)#.dropna()
    o=dfpiv_abs,dfpiv_freq
    with open(FN_DATA_CACHE_DEC,'wb') as of:
        pickle.dump(o,of)
    return o


def get_pathdf_models(period_len=5):
    pathdf=get_model_paths_df(PATH_MODELS_BPO, 'model.bin').sort_values(['period_start','run'])
    pathdf['period']=[f'{x}-{y}' for x,y in zip(pathdf.period_start, pathdf.period_end)]
    pathdf['period_len']=pathdf.period_end - pathdf.period_start
    if period_len: pathdf=pathdf[pathdf.period_len==period_len]
    return pathdf[~pathdf.period.isnull()]

VECNAMES=None
def get_vector_names():
    global VECNAMES
    if not VECNAMES: VECNAMES=list(pd.read_csv(FN_VECTOR_SCORES_DIFFMEANS).vector.unique())
    return VECNAMES

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


def get_dfchange_from_sattelzeit_models(words=None):
    # Load collective difference data
    dfchange=pd.read_csv(FN_CHANGE_RUNS_AVG).set_index('word').sort_values('rank')
    dfchange['class_abs']=dfchange.apply(classify_abstractness,1)
    dfchange['class_change']=dfchange.apply(classify_change,1)
    dfchange['class_signif']=[(x!='Abs~Conc' or y=='+Changed') for x,y in zip(dfchange.class_abs, dfchange.class_change)]
    dfchange['class']=[f'{x} {y}' for x,y in zip(dfchange.class_abs, dfchange.class_change)]
    
    # filter?
    return dfchange.sort_values('rank')
    #.query('class_abs!="Abs~Conc" | class_change=="+Changed"')
    # return dfchange

def get_dfchange_from_decade_data(fn=FN_VECTOR_SCORES_DIFFMEANS,wide=True):
    idf=pd.read_csv(fn).set_index(['vector','word'])
    if not wide: return idf
    df=None
    for vecname,gdf in idf.groupby('vector'):
        gdf=gdf.reset_index().drop('vector',1).set_index('word')
        gdf.columns=[c+'_'+vecname for c in gdf.columns]
        df=gdf if df is None else df.join(gdf)
    
    for c in df.columns:
        if c.startswith('mean_diff_t_abs'):
            df['perc_'+c]=df[c].apply(lambda x: percentileofscore(df[c],x))
    
    return df

def get_dfchange(words=None):
    odf=get_dfchange_from_sattelzeit_models().join(get_dfchange_from_decade_data(), how='outer')
    if words: odf=odf.reset_index()[odf.reset_index().word.isin(set(words))].set_index('word')
    return odf

def get_dfchange_simple(dfchange=None, col='mean_diff'):
    if dfchange is None: dfchange=get_dfchange()
    vecs=get_vector_names()
    odf=dfchange[[
        c
        for c in dfchange.columns
        if c.startswith(col+'_')
        and c[len(col)+1:] in set(vecs)
    ]]
    odf.columns=[c[len(col)+1:] for c in odf.columns]
    return odf


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
def to_z(pivdf,axis=1,progress=False):
    pivdf=pivdf.T if not axis else pivdf
    pivdfz=pd.DataFrame(index=pivdf.index, columns=pivdf.columns)
    for c in (tqdm(pivdf.columns) if progress else pivdf.columns):
        pivdfz[c]=(pivdf[c] - pivdf[c].mean()) / pivdf[c].std()
    return pivdfz.T if not axis else pivdfz


def start_fig(data=None, theme='minimal',text_size=8, figure_size=(8,8), **aesd):
    p9.options.figure_size=figure_size
    p9.options.dpi=600
    fig=p9.ggplot(p9.aes(**aesd), data=data)
    fig+=getattr(p9,f'theme_{theme}')()
    fig+=p9.theme(
        text=p9.element_text(size=text_size),
        plot_background=p9.element_rect(fill='white')
    )
    return fig
    
    
    
    

def get_rate_of_change_data(key='dist_local'):
    df=pd.read_pickle('data/data.semantic_change_over_decades.1run.v3.pkl')
    df['period']=[f'{x[:4]}s->{y[:4]}s' for x,y in zip(df.period1,df.period2)]
    df['period_int1']=[int(y[:4]) for y in df.period1]
    df['period_int2']=[int(y[:4]) for y in df.period2]    
    return df

def get_figdf1_rateofchange(df=None,randomize=False):
    if df is None: df=get_rate_of_change_data()
    pdf=df.groupby(['period','period1','period2']).mean().reset_index()
    if randomize:
        for c in ['period1','period2']:
            pdf[c]=list(pdf[c].sample(frac=1))
    pdf2=pd.DataFrame(pdf).rename(columns=dict(
        period1='period2',
        period2='period1',
    ))
    figdf=pdf.append(pdf2)
    figdf['period_int1']=figdf.period1.apply(lambda x: int(x[:4]))
    figdf['period_int2']=figdf.period2.apply(lambda x: int(x[:4]))
    figdf['perc_local_int']=figdf.dist_local.apply(lambda x: percentileofscore(figdf.dist_local, x)).apply(int)    
    return figdf



DFWAS=None
def get_word_abstractness_scores(cols=['Abs-Conc.Median.C18','Abs-Conc.Median.C19']):
    global DFWAS
    if DFWAS is None:
        DFWAS=get_allnorms()[cols].reset_index().dropna().set_index('word').mean(axis=1)
    return DFWAS


def prdz(y,ystart=1710,yend=3000,ystep=40):
    ln=None
    for n in range(ystart,yend,ystep):
        if ln is None: ln=n
        if y<n: return ln
        ln=n
        


def get_novelty_data(ifn=FN_NOVELTY_DATA):
    allres = pd.read_pickle(ifn).query('foote_novelty!=0.0')
    allres['is_signif']=[int(x<0.05 or y<0.05)
                        for x,y in zip(allres.p_peak,allres.p_trough)]
    allres['foote_size']=allres.foote_size.apply(int)
    allres['year']=allres.year.apply(int)
    allres = pd.concat(grp.assign(glen=len(grp)) for i,grp in allres.groupby(['foote_size','year'])).reset_index()
    allres = pd.concat(
        grp.sort_values('year').assign(
            foote_novelty_z=((grp.foote_novelty - grp.foote_novelty.dropna().mean()) / grp.foote_novelty.dropna().std())
        )#.set_index('year').rolling(rolling,min_periods=min_periods).mean()
        for i,grp in allres.groupby('foote_size')
    )
    return allres


C=get_corpus()
logger.remove()
logger.add(sys.stderr, format="{message}", filter='koselleck', level="INFO")
# logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
def log(*x,**y): logger.info(' '.join(str(xx) for xx in x),**y)
    

    
    
    
UPROOT='/Markdown/Drafts/TheGreatAbstraction/figures/'
def upfig(fnfn,uproot=UPROOT):
    ofnfn=os.path.join(uproot,os.path.basename(fnfn))
    cmd=f'dbu upload {fnfn} {ofnfn}'
    os.system(cmd)
    cmd = f'dbu share {ofnfn}'
    os.system(cmd)
    
    
    
    
def rsync(ifnfn,ofnfn,flags='-avP'):
    return runcmd(f'rsync {flags} {ifnfn} {ofnfn}')

def _rsync_data_from_ember(obj): return rsync(obj[0], obj[1])
def rsync_data_from_ember(num_proc=1):
    paths_I_want = [
        os.path.join(
            os.path.dirname(path),
            'dists.pkl'
        ).split('/ryan/')[-1] for path in get_model_paths_df(PATH_MODELS_BPO).path
    ]
    objs = [
        (f'ryan@ember:{fn}', os.path.join(os.path.expanduser('~'), fn))
        for fn in paths_I_want
    ]
    objs = [(x,y) for x,y in objs if not os.path.exists(y)]
    
    return pmap(_rsync_data_from_ember, objs, num_proc=num_proc, desc='Rsyncing data from ember')
def runcmd(cmd,verbose=False):
    import subprocess
    print('>>',cmd)
    result = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT).decode()
    return result


def periodize(y,ybin=5):
    y1=y//ybin*ybin
    y2=y1+ybin
    return f'{y1}-{y2}'