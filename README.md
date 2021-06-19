# koselleck
Computing Koselleck book chapter


## Workflow

1. [Models](koselleck/models.ipynb): Generate word embedding models
    a. Generate skipgrams
        * Done with LLTK elsewhere
    b. Generate models
        
        def gen_models(
                ybin=5,
                ymin=1680,
                ymax=1970,
                num_runs=1,
                force=False,
                nskip_per_yr=NSKIP_PER_YR
            )
    
    * Prominent divisions:
        * Half-decade: Used for semantic distance matrices and novelty data
        * 20-Years: Used for one version of the neighborhood plots
        * 40-years: used for another neighborhood plots

2. ...