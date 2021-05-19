from koselleck.imports import *


PLOT_CHANGE_ARROW_QUERY_STR='is_clean_noiseaware>0 & class_abs!="Abs~Conc" & ((score1_abstractness>1 & score2_abstractness<1) | (score1_abstractness<1 & score2_abstractness>1)) '

def plot_change_arrows(
        dfchange=None,
        query_str=PLOT_CHANGE_ARROW_QUERY_STR,):
    if dfchange is None: dfchange=get_dfchange()
    figdf=dfchange.reset_index().query(PLOT_CHANGE_ARROW_QUERY_STR)
    idvars=[c for c in figdf.columns if not c.startswith('score1') and not c.startswith('score2')]
    valvars=[c for c in figdf.columns if c.startswith('score1') or c.startswith('score2')]
    
    figdf=figdf.melt(id_vars=['word','perc_local','score_diff_abstractness','perc_abstractness','is_clean_noiseaware','class_abs'],
                    value_vars=['score1_abstractness','score2_abstractness'])
    figdf=figdf.sort_values(['word','variable'], ascending=[1,1])
    # figdf=pd.concat(g[1] for i,g in enumerate(figdf.groupby('word')) if len(g)==2)
    figdf


def iplot_word_info(w='culture',manual=False):
    roller=widgets.IntSlider(value=1,min=1,max=5)
    if manual:
        my_interact_manual = interact_manual.options(manual_name="Plot")
        im=my_interact_manual(
            plot_word_info,
            w=w,
            rolling=roller,
        )
        return im
    interact(plot_word_info,w=w,rolling=roller,title=fixed(''))

def classify_vector_change(row,perc_threshold=75):
    name=row.vector
    ns=str(name).split('.')[0].split('-')
    n1,n2=ns[0],ns[-1]
    res=f'{n1}~{n2}'
    if row.mean_diff_p<0.05 and row.perc>perc_threshold:
        res=f'+{n1}' if row.mean_diff>0 else (f'+{n2}' if n2!=n1 else f'-{n1}')
#         new=(row.mean2>1 and row.mean1<1) if row.mean_diff>0 else (row.mean2<-1 and row.mean1>-1)
        new=(row.mean2>1 and row.mean1<0.75) if row.mean_diff>0 else (row.mean2<-1 and row.mean1>-0.75)
        if new: res=res+'+'
    return res

def show_changes_for_word(words=[],only_signif=False):
    words=[words] if type(words)==str else words
    dfcc=get_classed_changes()
    if only_signif: dfcc=dfcc.query('mean_diff_p<0.05')
    cols=['word','vector','change','mean_diff','mean1','mean2','mean_diff_t','mean_diff_p','perc']#,'change_rank','mean_diff_abs']
    odf=dfcc[(dfcc.word.isin(set(words))) & ~(dfcc.vector.isna())].sort_values('perc',ascending=False)
    odf=round(odf,2).sort_values(['change_rank','mean_diff_abs'],ascending=False)
    odf=odf[cols].set_index(['word','vector'])
    odf['mean_diff']=[f'+{x}' if x>0 else x for x in odf.mean_diff]
    return odf

DFCC=None
def plot_word_info(w,rolling=2,title=''):
    

    # load data
    vectors=get_vector_names()
    vectors = [v for v in vectors if not v.endswith('.S') or v[:-2]+'.M' not in set(vectors)]
    dfchange,dfruns,dfpiv_abs,dfpiv_freq,dfruns_dec,dfpiv_ambig = get_df_package()
    dfcc=get_classed_changes()

    # format data for figures
    figdf=pd.DataFrame()#{'freq':dfpiv_freq.loc[w], 'ambig':dfpiv_ambig.loc[w]})
    try:
        for vec in vectors:
            figdf[vec]=get_dfpiv(vec, df=dfruns_dec).loc[w]
        figdf=figdf.rolling(rolling,min_periods=1).mean().rename_axis('year')
        figdf=figdf.reset_index().melt(id_vars=['year'])#,value_vars=['abs','freq'])
        # boxplot 1
        # figdf2=dfruns.loc[w].melt(value_vars=['score1_abstractness','score2_abstractness'], var_name='score_type')
        # figdf2['year']=figdf2.score_type.apply(lambda x: 1750 if '1' in x else 1850)
        # figdf2['variable']='Abs-Conc.Median'
        # boxplot 2
        figdf3=dfruns_dec.reset_index().set_index('word').loc[w]
        figdf3['year']=figdf3.period.apply(lambda x: int(x[:4]))
        figdf3=figdf3.query('1700<=year<1900')
        figdf3=figdf3.drop(['corpus','period','run'],1).melt(id_vars=['year'])
        # figdf3['score_type']=figdf3.year.apply(periodize_sattelzeit_binary)
        figdf3['score_type']=figdf3.year.apply(periodize_sattelzeit)
        figdf3=figdf3[~figdf3.score_type.isnull()]
        figdf3=figdf3[figdf3.variable.isin(set(vectors))]

        # for fdf in [figdf,figdf2,figdf3]:
        for fdf in [figdf,figdf3]:
            fdf['variable']=fdf['variable'].apply(lambda x: x.split('.')[0])
    except KeyError:
        return

    # create figure
    p9.options.figure_size=8,3
    p9.options.dpi=150
    fig=p9.ggplot(figdf, p9.aes(x='year',y='value'))#,color='variable'))
    fig+=p9.geom_line(alpha=1, color='black', size=0.25)
    fig+=p9.geom_boxplot(p9.aes(group='score_type'),size=0.35,data=figdf3,width=25, color='black', outlier_size=0.05, alpha=0.5,outlier_alpha=0.5)
    # fig+=p9.geom_boxplot(p9.aes(group='score_type'),size=0.35,data=figdf2,width=25, color='blue', outlier_size=0.05, alpha=0.5,outlier_alpha=0.5)
    # fig+=p9.geom_point(data=figdf2,size=.25,color='blue')
    fig+=p9.geom_point(size=0.5,alpha=0.5)
    fig+=p9.facet_wrap('variable',scales='free_y',nrow=3)
    fig+=p9.theme_minimal()
    fig+=p9.geom_hline(yintercept=0)
    fig+=p9.geom_hline(yintercept=-1,alpha=0)
    fig+=p9.geom_hline(yintercept=1,alpha=0)
    fig+=p9.labs(title=f'Semantic changes in the word "{w}" over 1700-1900', y='Normalized score', x='Decade-by-decade semantic models')
    fig+=p9.theme(text=p9.element_text(size=7), axis_text=p9.element_text(size=3))

    # * **{int(row.perc_abstractness)}%** (**{row.class_abs}**) percentile for magnitude change in <u>abstractness</u>** (z={round(row.dist_abstractness,2)}): {round(row.score1_abstractness,2)} -> {round(row.score2_abstractness,2)}"""
    # format markdown
    def neighb(nstr):
        if type(nstr)!=str: return ''
        return ' '.join(
            f'<u>**{x}**</u>' if not x[0].isalpha() and x[0]!='(' else x
            for x in nstr.split()
        )
    omd=f"""
## {w if not title else title}"""
    try:
        row=dfchange.loc[w]
        omd+=f"""
* Data from Sattelzeit binary model comparison
    * **{"Clean" if row.is_clean_noiseaware else "Noisy"}** result for semantic change according to the noise aware data
    * **{int(row.perc_local)}%** (**{row.class_change}**) percentile for <u>local semantic change</u> (z={round(row.dist_local,2)})"""
    except (ValueError,KeyError) as e:
        pass


#     omd+="""
# * Data from decade-by-decade models"""
#     prefix='perc_mean_diff_t_abs_'
#     ckeys=[ck for ck in row.keys() if ck.startswith(prefix)]
#     ckeys.sort(key=lambda ck: -row[ck])

#     for c in ckeys:
#         name=c.split(prefix)[-1]
#         if not name in set(vectors): continue
#         p=row['mean_diff_p_'+name]
#         z=row['mean_diff_'+name]
#         ns=name.split('.')[0].split('-')
#         n1,n2=ns[0],ns[-1]
#         res='~'#f'{n1}~{n2}'
#         signif=''
#         if p<0.05:
#             res=f'+{n1}' if z>0 else (f'+{n2}' if n2!=n1 else f'-{n1}')
#             mean1,mean2,mdiff=row['mean1_'+name],row['mean2_'+name],row['mean_diff_'+name]
#             new=(mean2>1 and mean1<.75) if mdiff>0 else (mean2<-1 and mean1>-.75)
#             if new:
#                 res='+'+res
#                 signif='***'
#             elif (abs(z)>1 or row[c]>=75):
#                 signif='**'
#         b=signif
# # * {b}{int(row[c])}%{b} ({b}{res}{b}) magnitude change in **{name}** (t={round(row["mean_diff_t_"+name],2)}): {round(row["mean1_"+name],2)} -> {round(row["mean2_"+name],2)}"""
#         try:
#             omd+=f"""
#     * {b}{int(row[c])}%{b} ({b}{res}{b}) magnitude change in <u>{name}</u> ({"+" if row["mean_diff_"+name]>0 else ""}{round(row["mean_diff_"+name],2)}): {round(row["mean1_"+name],2)} -> {round(row["mean2_"+name],2)}"""
#         except ValueError:
#             pass

    omd+=f"""
* Its neighborhoods:

| Vor der Sattelzeit (1700-1770) | Nach der Sattelzeit (1830-1900) |
| -------------- | -------------- |
| {neighb(dfchange.loc[w].neighborhood1_local)} | {neighb(dfchange.loc[w].neighborhood2_local)} |

* Its changes along key vectors categorized:
    """
    printm(omd)
    display(show_changes_for_word(w))
    printm('* These changes visualized:')
    display(fig)