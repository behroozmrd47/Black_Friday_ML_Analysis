import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analysis_cat_dependency(dataframe, var_v, var_h, is_plot=True):
    df2 = dataframe.groupby([var_v, var_h]).size().reset_index(name='Count')
    df_count = pd.pivot_table(df2, values='Count', columns=[var_v], index=[var_h], aggfunc=np.sum)
    df_ratio_val_1 = df_count.div(df_count.sum(axis=0), axis=1)
    df_ratio_val_2 = df_count.div(df_count.sum(axis=1), axis=0)
    if is_plot:
        df_ratio_val_1.plot(kind='bar')
        plt.show()
        df_ratio_val_2.plot(kind='bar')
        plt.show()

    from scipy.stats import chi2_contingency
    chi, pval, dof, exp = chi2_contingency(df_count)
    significance = 0.05
    print('p-value=%.6f, significance=%.2f\n' % (pval, significance))
    if pval < significance:
        print("""At %.2f level of significance, we reject the null hypotheses and accept H1.
    They are not independent.""" % significance)
    else:
        print("""At %.2f level of significance, we accept the null hypotheses.
    They are independent.""" % significance)
    return df_count, exp
