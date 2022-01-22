"""
Created on Thu May  7 13:09:16 2020

@author: vasanthan.s
"""
import pandas as pd
import numpy as np


def info_val(y, inp_fe, in_df, out_df):
    """ Calculate Weight of evidence and information value for train and test/out of time data 
    with variables kept at same level. Used for Binary classification only

    Args:
        y (string): Name of the Y variable
        inp_fe (list): List containg the list of features for which the metrics should be calculated
        in_df (DataFrame): Train DataFrame
        out_df (DataFrame): Test/Out of time DataFrame

    Returns:
        DataFrames: DataFrames with WOE and IV for train and Test/Out of time data
    """
    iv_df1_in = pd.DataFrame()
    iv_df2_in = pd.DataFrame()
    iv_df1_out = pd.DataFrame()
    iv_df2_out = pd.DataFrame()
    k = 0
    # Loop for every features present in the code
    for i in inp_fe:
        print(i)
        k = k+1
        # Check wether the variable is not Y variable itself
        if i != y:
            # Check the data type of the column and accordingly get count and #y instances for each value/bin
            if in_df[i].dtypes == 'object':
                #                 print(" in if ")
                in_df[i] = in_df[i].fillna("NULL")
                out_df[i] = out_df[i].fillna("NULL")
                temp_in = in_df.groupby(in_df[i])[y].count()
                temp1_in = in_df.groupby(in_df[i])[y].sum()
                temp_out = out_df.groupby(out_df[i])[y].count()
                temp1_out = out_df.groupby(out_df[i])[y].sum()
            else:
                # filling NA with arbitary negative number . This helps in binning
                in_df[i] = in_df[i].fillna(-999999999.0)
                out_df[i] = out_df[i].fillna(-999999999.0)

                # Bin the continuous variable into 10 bins. Change the number of bins as required
                q = pd.qcut(in_df[i], 10, retbins=True, duplicates='drop')
                bin_vals = q[1].tolist()
                if(-999999999.0 in in_df[i].unique()):
                    bin_vals.insert(0, -999999999)
                    bin_vals[0] = float(bin_vals[0]-1)
                else:
                    bin_vals[0] = float(bin_vals[0]-1)
                seen = set()
                bin_vals = [x for x in bin_vals if not (
                    x in seen or seen.add(x))]
                in_df[i+'_bin'] = pd.DataFrame(pd.cut(in_df[i],
                                               bins=bin_vals, duplicates='drop'))
                temp_in = in_df.groupby(in_df[i+'_bin'])[y].count()
                temp1_in = in_df.groupby(in_df[i+'_bin'])[y].sum()
                out_df[i+'_bin'] = pd.DataFrame(
                    pd.cut(out_df[i], bins=bin_vals, duplicates='drop'))
                t = out_df[i+'_bin'].dropna().max()
                out_df[i+'_bin'].fillna(t, inplace=True)
                temp_out = out_df.groupby(out_df[i+'_bin'])[y].count()
                temp1_out = out_df.groupby(out_df[i+'_bin'])[y].sum()
            # Calculating WOE and IV for intime data
            iv_df = pd.DataFrame(
                {'Levels': temp_in.index, 'total': temp_in.values})
            iv_df['keys'] = i
            iv_df['perc_in_feature'] = (iv_df['total']/iv_df['total'].sum())

            iv_df['Target_counts'] = temp1_in.values
            iv_df['non_Target_counts'] = iv_df['total']-iv_df['Target_counts']
            iv_df['lower'] = iv_df['Levels'].apply(
                lambda x: x.left if type(x) != str else x)
            iv_df['upper'] = iv_df['Levels'].apply(
                lambda x: x.right if type(x) != str else x)

            iv_df['Target_perc_current_level'] = (
                iv_df['Target_counts']/(iv_df['non_Target_counts']+iv_df['Target_counts']))
            iv_df['Target_percent_wrt_total'] = (
                iv_df['Target_counts']/iv_df['Target_counts'].sum())

            iv_df['Non_Target_perc_current_level'] = (
                iv_df['non_Target_counts']/(iv_df['non_Target_counts']+iv_df['Target_counts']))
            iv_df['Non_Target_percent_wrt_total'] = (
                iv_df['non_Target_counts']/iv_df['non_Target_counts'].sum())

            iv_df['WOE'] = np.log(
                iv_df['Non_Target_percent_wrt_total']/iv_df['Target_percent_wrt_total'])
            iv_df['IV'] = ((iv_df['Non_Target_percent_wrt_total'] -
                           iv_df['Target_percent_wrt_total']) * iv_df['WOE'])
            iv_df = iv_df[['keys', 'Levels', 'lower', 'upper', 'total', 'perc_in_feature', 'Target_counts', 'non_Target_counts',
                           'Target_perc_current_level', 'Target_percent_wrt_total',
                          'Non_Target_perc_current_level', 'Non_Target_percent_wrt_total',
                           'WOE', 'IV']]
            iv_df.columns = [str(i)+"_intime" for i in iv_df.columns]
            t = iv_df.append(pd.Series(name=len(iv_df)))
            iv_df1_in = pd.concat([iv_df1_in, iv_df], axis=0)
            iv_df2_in = pd.concat([iv_df2_in, t], axis=0)
            # Calculating WOE and IV for outtime data
            iv_df_out = pd.DataFrame(
                {'Levels': temp_out.index, 'total': temp_out.values})
            iv_df_out['keys'] = i
            iv_df_out['perc_in_feature'] = (
                iv_df_out['total']/iv_df_out['total'].sum())
            iv_df_out['Target_counts'] = temp1_out.values
            iv_df_out['non_Target_counts'] = iv_df_out['total'] - \
                iv_df_out['Target_counts']
            iv_df_out['lower'] = iv_df_out['Levels'].apply(
                lambda x: x.left if type(x) != str else x)
            iv_df_out['upper'] = iv_df_out['Levels'].apply(
                lambda x: x.right if type(x) != str else x)

            iv_df_out['Target_perc_current_level'] = (
                iv_df_out['Target_counts']/(iv_df_out['non_Target_counts']+iv_df_out['Target_counts']))
            iv_df_out['Target_percent_wrt_total'] = (
                iv_df_out['Target_counts']/iv_df_out['Target_counts'].sum())

            iv_df_out['Non_Target_perc_current_level'] = (
                iv_df_out['non_Target_counts']/(iv_df_out['non_Target_counts']+iv_df_out['Target_counts']))
            iv_df_out['Non_Target_percent_wrt_total'] = (
                iv_df_out['non_Target_counts']/iv_df_out['non_Target_counts'].sum())

            iv_df_out['WOE'] = np.log(
                iv_df_out['Non_Target_percent_wrt_total']/iv_df_out['Target_percent_wrt_total'])
            iv_df_out['IV'] = ((iv_df_out['Non_Target_percent_wrt_total'] -
                               iv_df_out['Target_percent_wrt_total']) * iv_df_out['WOE'])

            iv_df_out = iv_df_out[['keys', 'Levels', 'lower', 'upper', 'total', 'perc_in_feature', 'Target_counts', 'non_Target_counts',
                                  'Target_perc_current_level', 'Target_percent_wrt_total',
                                   'Non_Target_perc_current_level', 'Non_Target_percent_wrt_total',
                                   'WOE', 'IV']]
            iv_df_out.columns = [str(i)+"_outtime" for i in iv_df_out.columns]
            t = iv_df_out.append(pd.Series(name=len(iv_df_out)))
            iv_df1_out = pd.concat([iv_df1_out, iv_df_out], axis=0)
            iv_df2_out = pd.concat([iv_df2_out, t], axis=0)

    return iv_df1_in, iv_df1_out
