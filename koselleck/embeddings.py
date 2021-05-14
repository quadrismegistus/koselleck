from .imports import *


"""
Semantic change functions
"""

def get_shared_vocab(m1,m2,words=None):
    shared=set(m1.wv.key_to_index.keys()) & set(m2.wv.key_to_index.keys())
    shared=set(shared)&set(words) if words else shared
    shared=list(shared)
    shared.sort()
    return shared

def measure_change_noiseaware(m1,m2,words=None,num_proc=1,progress=True,**pmap_attrs):
    shared=get_shared_vocab(m1,m2,words=words)
    
    # get matrices
    X=pd.DataFrame([m1.wv.vectors[m1.wv.key_to_index[k]] for k in shared], index=shared)
    Y=pd.DataFrame([m2.wv.vectors[m2.wv.key_to_index[k]] for k in shared], index=shared)
    
    # align
    transform_matrix, alpha, clean_indices, noisy_indices = noise_aware(X.values,Y.values,progress=progress)
    
    # objs
    objs=[(w,i,X.loc[w,:],Y.loc[w,:],transform_matrix,clean_indices) for i,w in enumerate(shared)]
    return pd.DataFrame(
        pmap_iter(
            do_measure_change_noiseaware,
            objs,
            num_proc=num_proc,
            desc='Calculating noise aware distances',
            progress=progress,
            **pmap_attrs
        )
    ).set_index('word')

# get similarities
def do_measure_change_noiseaware(obj):
    w,i,x,y,transform_matrix,clean_indices = obj
    dist =  cosine(np.dot(x,transform_matrix),y)
    return {
        'word':w,
        'dist':dist,
        'is_clean':i in set(clean_indices)
    }


def measure_change_procrustes(m1,m2,words=None,progress=False,**pmap_attrs):
    shared=get_shared_vocab(m1,m2,words)
    m1df,m2df=smart_procrustes_align_gensim_df(m1,m2,words=shared)
    return pd.DataFrame([
        {
            'word':w,
            'dist':cosine(m1df.loc[w], m2df.loc[w])
        }
        for w in (tqdm(shared,desc='Calculating procrustes distances',position=0) if progress else shared)
    ]).set_index('word').sort_values('dist')


def measure_change_local(model1,model2,words=None,k=50,num_proc=1,**pmap_attrs):
    """
    Basic implementation of William Hamilton (@williamleif) et al's measure of semantic change
    proposed in their paper "Cultural Shift or Linguistic Drift?" (https://arxiv.org/abs/1606.02821),
    which they call the "local neighborhood measure." They find this measure better suited to understand
    the semantic change of nouns owing to "cultural shift," or changes in meaning "local" to that word,
    rather than global changes in language ("linguistic drift") use that are better suited to a
    Procrustes-alignment method (also described in the same paper.)

    Arguments are:
    - `model1`, `model2`: Are gensim word2vec models.
    - `word` is a sting representation of a given word.
    - `k` is the size of the word's neighborhood (# of its closest words in its vector space).
    """
    shared=set(model1.wv.key_to_index.keys()) & set(model2.wv.key_to_index.keys())
    shared_f=shared&set(words) if words else shared
    # objs = [(word,model1,model2,k,shared_f) for word in shared_f]
    objs = [
        (
            word,
            model1,
            model2,
            k,
            shared_f
        ) for word in shared_f
    ]
    return pd.DataFrame(
        pmap_iter(
            do_measure_change_local,
            objs,
            num_proc=num_proc,
            desc='Calculating local neighborhood distances',
            **pmap_attrs
        )
    ).set_index('word').sort_values('dist',ascending=False)

def do_measure_change_local(obj):
    word,model1,model2,k,shared=obj
    # Get the two neighborhoods
    neighborhood1 = [w for w,c in model1.wv.most_similar(word,topn=k*10) if w in shared][:k]
    neighborhood2 = [w for w,c in model2.wv.most_similar(word,topn=k*10) if w in shared][:k]
    # Get the 'meta' neighborhood (both combined)
    meta_neighborhood = list(set(neighborhood1)|set(neighborhood2))
    # Filter the meta neighborhood so that it contains only words present in both models
    meta_neighborhood = [w for w in meta_neighborhood if w in shared]

    # For both models, get a similarity vector between the focus word and all of the words in the meta neighborhood
    vector1=np.array([fastdist.cosine(model1.wv[word], model1.wv[w]) for w in meta_neighborhood])
    vector2=np.array([fastdist.cosine(model2.wv[word], model2.wv[w]) for w in meta_neighborhood])
    # Compute the cosine distance *between* those similarity vectors
    dist=fastdist.cosine(vector1,vector2)
    return {
        'word':word,
        'dist':dist,
        'neighborhood1':', '.join([
            '-'+w if w not in set(neighborhood2) else w
            for w in neighborhood1
        ]),
        'neighborhood2':', '.join([
            '+'+w if w not in set(neighborhood1) else w
            for w in neighborhood2
        ]),
    }


def do_measure_change_contrast(obj,progress=True):
    m1,m2,contrastd,words=obj
    vec1=compute_vector(m1,contrastd['pos'],contrastd['neg'])
    vec2=compute_vector(m2,contrastd['pos'],contrastd['neg'])
    odf=pd.DataFrame([
        {
            'word':w,
            'score1':cosine(m1.wv[w], vec1),
            'score2':cosine(m2.wv[w], vec2),
            **dict((k,v) for k,v in contrastd.items() if type(v) in {int,float,str})
        }
        for w in (tqdm(words,desc='Measuring change in abstractness',position=0) if progress else words)
    ])
    odf['score1']=zscore(odf['score1'])
    odf['score2']=zscore(odf['score2'])
    odf['score_diff']=odf['score2']-odf['score1']
    odf['dist']=zscore(odf['score_diff'].apply(abs))
    return odf

def measure_change_abstractness(m1,m2,words=None,num_proc=1,sources={'Median'},**y):
    shared=get_shared_vocab(m1,m2,words)
    contrasts=get_origcontrasts()
    objs = [(m1,m2,contrastd,shared) for contrastd in contrasts
           if not sources or contrastd['source'] in sources]
    return pd.concat(
        pmap_iter(
            do_measure_change_contrast,
            objs,
            num_proc=num_proc,
            progress=False,
            desc='Measuring change in abstractness',
            kwargs=y
        )
    ).sort_values('score_diff').set_index('word')
        
    


def measure_change(
        m1,
        m2,
        words=None,
        funcs = [
            measure_change_abstractness,
            measure_change_noiseaware, 
            measure_change_procrustes,
            measure_change_local,
        ],
        **attrs):
    df=None
    for func in funcs:
        suffix=func.__name__.split('_')[-1]
        fdf=func(m1,m2,words=words,**attrs)
        if 'dist' in set(fdf.columns):
            fdf['dist']=zscore(fdf['dist'])
            fdf['rank']=(1-fdf['dist']).rank()
            fdf['perc']=fdf['dist'].apply(lambda x: percentileofscore(fdf.dist, x))
            # fdf['z']=zscore(fdf['dist'])
        fdf.columns=[f'{c}_{suffix}' for c in fdf.columns]
        df=df.join(fdf) if df is not None else fdf
    dist_cols = [c for c in df.columns if c.startswith('dist_')]
    crank_cols=[c for c in df.columns if c.startswith('rank_')]
    perc_cols=[c for c in df.columns if c.startswith('perc_')]
    neighb_cols = [c for c in df.columns if c.startswith('neighb')]
    df['dist']=df[dist_cols].mean(axis=1)
    df['rank']=df[crank_cols].mean(axis=1)
    df['perc']=df[perc_cols].mean(axis=1)
    prefixcols=['rank','perc','dist']
    other_cols = [c for c in df.columns if c not in set(dist_cols)|set(crank_cols)|set(neighb_cols)|set(prefixcols)|set(perc_cols)]
    return df[prefixcols + neighb_cols + crank_cols + perc_cols + dist_cols + other_cols].sort_values('rank')
    



"""
Alignment functions
"""



def smart_procrustes_align_gensim_df(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """
    # make sure vocabulary and indices are aligned
    shared = get_shared_vocab(base_embed,other_embed,words=words)

    # get the (normalized) embedding matrices
    base_vecs = base_embed.wv.get_normed_vectors()
    other_vecs = other_embed.wv.get_normed_vectors()

    # filter
    base_vecs = np.array([base_vecs[base_embed.wv.key_to_index[w]] for w in shared])
    other_vecs = np.array([other_vecs[other_embed.wv.key_to_index[w]] for w in shared])
    
    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    #other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    other_vecs_aligned = other_vecs.dot(ortho)
    return pd.DataFrame(base_vecs, index=shared), pd.DataFrame(other_vecs_aligned,index=shared)
#     return (base_embed,other_embed)





"""
Utilities
"""


def get_neighbors(w,m1,m2=None,topn=25,wide=True):
    ms=[m1,m2] if m2 is not None else [m1]
    df=pd.DataFrame([
        {
            'word1':w,
            'word2':w2,
            'model':mi,
            'csim':csim,
            'crank':wi
        }
        for mi,m in enumerate(ms)
        for wi,(w2,csim) in enumerate(m.wv.most_similar(w,topn=topn))
    ])
    if not wide:
        return df
    else:
        odf=None
        for i,gdf in sorted(df.groupby('model')):
            gdf=gdf.set_index('crank').sort_index()
            if odf is None: odf=pd.DataFrame(index=gdf.index)
            odf[f'model_{i}']=gdf.word2
        return odf


def get_centroid(model,words):
    words=[words] if type(words)==str else words
    vectors=[]
    for w in words:
        if w in model.wv.key_to_index:
            vectors+=[model.wv[w]]
    if not vectors: return None
    return np.mean(vectors,0)

def compute_vector(model,words_pos=[],words_neg=[]):
    centroid_pos=get_centroid(model,words_pos)
    if not words_neg: return centroid_pos
    centroid_neg=get_centroid(model,words_neg)
    if centroid_neg is not None:
        return centroid_pos - centroid_neg
    else:
        return centroid_pos

def compute_vector_scores(m,pos,neg=None,z=True):
    vec=np.array(compute_vector(m,pos,neg), dtype=np.float64)
    matrix=np.array(m.wv.get_normed_vectors(),dtype=np.float64)
    res=fastdist.vector_to_matrix_distance(vec,matrix,fastdist.cosine,'cosine')
    resd=dict((m.wv.index_to_key[i],x)for i,x in enumerate(res))
    s=pd.Series(resd)
    if z: s=(s - s.mean())/s.std()
    return s.sort_values()







class SkipgramsSampler:
    def __init__(self, fn, num_skips_wanted=None):
            self.fn=fn
            self.num_skips_wanted=num_skips_wanted
            self.num_skips=self.get_num_lines()
            nskip=num_skips_wanted if num_skips_wanted and self.num_skips>num_skips_wanted else self.num_skips
            self.line_nums_wanted = set(random.sample(list(range(nskip)), nskip))

    def get_num_lines(self):
            then=time.time()
#               print('>> [SkipgramsSampler] counting lines in',self.fn)
            with gzip.open(self.fn,'rb') if self.fn.endswith('.gz') else open(self.fn) as f:
                    for i,line in enumerate(f):
                            pass
            num_lines=i+1
            now=time.time()
#               print('>> [SkipgramsSampler] finished counting lines in',self.fn,'in',int(now-then),'seconds. # lines =',num_lines,'and num skips wanted =',self.num_skips_wanted)
            return num_lines

    def __iter__(self):
            i=0
            with gzip.open(self.fn,'rb') if self.fn.endswith('.gz') else open(self.fn) as f:
                    for i,line in enumerate(f):
                            line = line.decode('utf-8') if self.fn.endswith('.gz') else line
                            if i in self.line_nums_wanted:
                                    yield line.strip().split()

class SkipgramsSamplers:
    def __init__(self,fns,nskip,**y):
        self.skippers=[SkipgramsSampler(fn,nskip) for fn in fns]
    def __iter__(self):
        for skipper in self.skippers:
            yield from skipper