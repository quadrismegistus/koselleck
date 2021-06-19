# koselleck
Computing Koselleck book chapter


## Workflow

### 1. [Models](koselleck/models.ipynb): Generate word embedding models

#### a. Generate skipgrams

* Done with LLTK elsewhere
#### b. Generate models

```py
def gen_models(
        ybin=5,
        ymin=1680,
        ymax=1970,
        num_runs=1,
        force=False,
        nskip_per_yr=NSKIP_PER_YR
    )
```

* Prominent divisions:
    * Half-decade: Used for semantic distance matrices and novelty data
    * 20-Years: Used for one version of the neighborhood plots
    * 40-years: used for another neighborhood plots

### 2. [Vecs](koselleck/vecs.ipynb): Saving vector spaces to db

* Function for loading and caching vectors:

```py
def vecs(period,run=1,corpus=DEFAULT_CORPUS,words=[]):
    qstr=get_vec_qstr(period,run)
    res=None
    with get_veclib('vecs',autocommit=False) as vl:
        if qstr in vl:
            res=vl[qstr]
        else:
            mpath=os.path.join(PATH_MODELS,corpus,period,f'run_{run:02}','model.bin')
            m=load_model(mpath)
            data=m.wv.vectors
            keys=[m.wv.index_to_key[i] for i in range(len(data))]
            res=pd.DataFrame(data, index=keys)
            vl[qstr]=res
            vl.commit()            
        if words: res=res.loc[[i for i in res.index if i in set(words)]]
    return res
```