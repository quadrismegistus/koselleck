from .imports import *

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

def iplot_word_info(w='culture'): return interact(plot_word_info,w=w)

def plot_word_info(w):
    
    try:
        dfchange,dfruns,dfpiv_abs,dfpiv_freq = get_df_package()

        p9.options.figure_size=3,2
        p9.options.dpi=150
        figdf2=dfruns.loc[w].melt(value_vars=['score1_abstractness','score2_abstractness'], var_name='score_type')
        figdf2['year']=figdf2.score_type.apply(lambda x: 1750 if '1' in x else 1850)
        figdf2['variable']='abs'
        
        
        figdf=pd.DataFrame({'abs':dfpiv_abs.loc[w], 'freq':dfpiv_freq.loc[w]})
        figdf=figdf.reset_index().melt(id_vars=['year'],value_vars=['abs','freq'])

        fig=p9.ggplot(figdf, p9.aes(x='year',y='value'))#,color='variable'))
        fig+=p9.geom_boxplot(p9.aes(group='score_type'),data=figdf2,width=50)
        fig+=p9.geom_point(data=figdf2,size=.25)
        fig+=p9.geom_point(size=0.5,alpha=0.5)
        fig+=p9.geom_line(alpha=0.5)
        fig+=p9.facet_wrap('variable',scales='free_y')
        fig+=p9.theme_classic()
        fig+=p9.geom_hline(yintercept=0)
        fig+=p9.geom_hline(yintercept=-1,alpha=0)
        fig+=p9.geom_hline(yintercept=1,alpha=0)
        fig+=p9.labs(title=f'Relative abstractness and frequency of {w} over 1700-1900')
        fig+=p9.theme(text=p9.element_text(size=7))

        def neighb(nstr):
            return ' '.join(
                f'<u>**{x}**</u>' if not x[0].isalpha() and x[0]!='(' else x
                for x in nstr.split()
            )
        row=dfchange.loc[w]
        omd=f"""
## {w}
* **{int(row.perc_local)}%** (**{row.class_change}**) percentile for local semantic change (z={round(row.dist_local,2)})
* **{int(row.perc_abstractness)}%** (**{row.class_abs}**) percentile for magnitude change in abstractness (z={round(row.dist_abstractness,2)}): {round(row.score1_abstractness,2)} -> {round(row.score2_abstractness,2)}
* **{"Clean" if row.is_clean_noiseaware else "Noisy"}** result according to the noise aware data
* Its neighborhoods:

| Neighborhood 1 | Neighborhood 2 |
| -------------- | -------------- |
| {neighb(dfchange.loc[w].neighborhood1_local)} | {neighb(dfchange.loc[w].neighborhood2_local)} |
        """
        printm(omd)
        
#         display(fig2)
        display(fig)
    except KeyError as e:
        # print('!!',e)
        pass