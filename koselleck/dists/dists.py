from koselleck.imports import *

def get_all_localdists():
    global DF_LOCALDISTS
    if DF_LOCALDISTS is None: DF_LOCALDISTS=get_cross_model_dists()
    #DF_LOCALDISTS=pd.read_pickle(FN_ALL_LOCALDISTS).groupby(['period1','period2','word']).mean()
    return DF_LOCALDISTS



def plot_distmat(distdf,xcol='period1',ycol='period2',value_name='value',use_color=False,xlim=None,ylim=None,title='Distance matrix',ofn=None,invert=False,**y):
    ddf=distdf
    
    
    distdfm=distdf.reset_index().melt(id_vars=[xcol],value_name=value_name).dropna()
#     display(distdfm)
    
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
        fill='Semantic distance (LNM)',
        title=title
    )
    if ofn:
        ofnfn=os.path.join(PATH_FIGS,ofn)
        fig.save(ofnfn)
        upfig(ofnfn)
        #fig.save(os.path.join('/home/ryan/Markdown/Drafts/TheGreatAbstraction/figures', ofn))
        
    return fig


def plot_historical_semantic_distance_matrix(words,save=False,vnum='v34',**y):
    wstr=words.strip() if type(words)==str else '-'.join(words)
    wstr2=words.strip() if type(words)==str else ', '.join(words)
    return plot_distmat(
        get_historical_semantic_distance_matrix(
            words,
            **y
        ),
        figure_size=(8,8),
        ofn=f'fig.distmat.{wstr}.{vnum}.png' if save else None,
        title=f'Historical-semantic distance matrix for ‘{wstr2}’',
        
    )