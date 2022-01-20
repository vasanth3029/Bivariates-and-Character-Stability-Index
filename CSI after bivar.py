# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:09:16 2020

@author: vasanthan.s
"""


def info_val(y,inp_fe,in_df,out_df):
    iv_df1_in=pd.DataFrame()
    iv_df2_in=pd.DataFrame()
    iv_df1_out=pd.DataFrame()
    iv_df2_out=pd.DataFrame()
    k=0
    for i in inp_fe:
        print(i)
        k=k+1
        if i!=y:
            if in_df[i].dtypes=='object':
#                 print(" in if ")
                in_df[i]=in_df[i].fillna("NULL")
                out_df[i]=out_df[i].fillna("NULL")
                temp_in=in_df.groupby(in_df[i])[y].count()
                temp1_in=in_df.groupby(in_df[i])[y].sum()
                temp_out=out_df.groupby(out_df[i])[y].count()
                temp1_out=out_df.groupby(out_df[i])[y].sum()
            else:
                in_df[i]=in_df[i].fillna(-999999999.0)
                out_df[i]=out_df[i].fillna(-999999999.0)
                q=pd.qcut(in_df[i],10,retbins=True,duplicates='drop')
                bin_vals=q[1].tolist()
                if(-999999999.0 in in_df[i].unique()):
                    bin_vals.insert(0,-999999999)
                    bin_vals[0]=float(bin_vals[0]-1)
                else:
                    bin_vals[0]=float(bin_vals[0]-1)
                seen = set()
                bin_vals= [x for x in bin_vals if not (x in seen or seen.add(x))]
                in_df[i+'_bin']=pd.DataFrame(pd.cut(in_df[i],bins=bin_vals,duplicates='drop'))
                temp_in=in_df.groupby(in_df[i+'_bin'])[y].count()
                temp1_in=in_df.groupby(in_df[i+'_bin'])[y].sum()
                out_df[i+'_bin']=pd.DataFrame(pd.cut(out_df[i],bins=bin_vals,duplicates='drop'))
                t=out_df[i+'_bin'].dropna().max()
                out_df[i+'_bin'].fillna(t,inplace=True)
                temp_out=out_df.groupby(out_df[i+'_bin'])[y].count()
                temp1_out=out_df.groupby(out_df[i+'_bin'])[y].sum()
            ## for intime
            iv_df=pd.DataFrame({'Levels':temp_in.index,'total':temp_in.values})
            iv_df['keys']=i
            iv_df['perc_in_feature']=(iv_df['total']/iv_df['total'].sum())
            iv_df['Target_counts']=temp1_in.values
            iv_df['non_Target_counts']=iv_df['total']-iv_df['Target_counts']
            iv_df['lower']=iv_df['Levels'].apply(lambda x:x.left if type(x)!=str else x)
            iv_df['upper']=iv_df['Levels'].apply(lambda x:x.right if type(x)!=str else x)
            iv_df['Target_perc']=(iv_df['Target_counts']/(iv_df['non_Target_counts']+iv_df['Target_counts']))
            iv_df['Target_count_percent_in_feature']=(iv_df['Target_counts']/iv_df['Target_counts'].sum())
            iv_df['non_Target_perc']=(iv_df['non_Target_counts']/(iv_df['non_Target_counts']+iv_df['Target_counts']))
            iv_df['non_Target_count_percent_in_feature']=(iv_df['non_Target_counts']/iv_df['non_Target_counts'].sum())
            iv_df['WOE']=np.log(iv_df['non_Target_count_percent_in_feature']/iv_df['Target_count_percent_in_feature'])
            iv_df['IV']=((iv_df['non_Target_count_percent_in_feature']-iv_df['Target_count_percent_in_feature'])* iv_df['WOE'])
            iv_df=iv_df[['keys','Levels','lower','upper','total','perc_in_feature','Target_counts','non_Target_counts',
                         'Target_perc','Target_count_percent_in_feature',
                         'non_Target_perc','non_Target_count_percent_in_feature',
                         'WOE','IV']]
            iv_df.columns=[str(i)+"_intime" for i in iv_df.columns]
            t=iv_df.append(pd.Series(name=len(iv_df)))
            iv_df1_in=pd.concat([iv_df1_in,iv_df],axis=0)
            iv_df2_in=pd.concat([iv_df2_in,t],axis=0)
            # for outtime
            iv_df_out=pd.DataFrame({'Levels':temp_out.index,'total':temp_out.values})
            iv_df_out['keys']=i
            iv_df_out['perc_in_feature']=(iv_df_out['total']/iv_df_out['total'].sum())
            iv_df_out['Target_counts']=temp1_out.values
            iv_df_out['non_Target_counts']=iv_df_out['total']-iv_df_out['Target_counts']
            iv_df_out['lower']=iv_df_out['Levels'].apply(lambda x:x.left if type(x)!=str else x)
            iv_df_out['upper']=iv_df_out['Levels'].apply(lambda x:x.right if type(x)!=str else x)
            iv_df_out['Target_perc']=(iv_df_out['Target_counts']/(iv_df_out['non_Target_counts']+iv_df_out['Target_counts']))
            iv_df_out['Target_count_percent_in_feature']=(iv_df_out['Target_counts']/iv_df_out['Target_counts'].sum())
            iv_df_out['non_Target_perc']=(iv_df_out['non_Target_counts']/(iv_df_out['non_Target_counts']+iv_df_out['Target_counts']))
            iv_df_out['non_Target_count_percent_in_feature']=(iv_df_out['non_Target_counts']/iv_df_out['non_Target_counts'].sum())
            iv_df_out['WOE']=np.log(iv_df_out['non_Target_count_percent_in_feature']/iv_df_out['Target_count_percent_in_feature'])
            iv_df_out['IV']=((iv_df_out['non_Target_count_percent_in_feature']-iv_df_out['Target_count_percent_in_feature'])* iv_df_out['WOE'])
            iv_df_out=iv_df_out[['keys','Levels','lower','upper','total','perc_in_feature','Target_counts','non_Target_counts',
                         'Target_perc','Target_count_percent_in_feature',
                         'non_Target_perc','non_Target_count_percent_in_feature',
                         'WOE','IV']]
            iv_df_out.columns=[str(i)+"_outtime" for i in iv_df_out.columns]
            t=iv_df_out.append(pd.Series(name=len(iv_df_out)))
            iv_df1_out=pd.concat([iv_df1_out,iv_df_out],axis=0)
            iv_df2_out=pd.concat([iv_df2_out,t],axis=0)
            
    return iv_df1_in,iv_df1_out


    #biavr=pd.merge(iv_df1_in,iv_df1_out,how='outer',left_on=['keys_intime','Levels_intime'],right_on=['keys_outtime','Levels_outtime'])
    #bivar_space=pd.merge(iv_df2_in,iv_df2_out,how='outer',left_on=['keys_intime','Levels_intime'],right_on=['keys_outtime','Levels_outtime'])
int_bivar,out_bivar=info_val(y,train.columns,train.copy(),x_oot.copy())
intime=int_bivar[['keys_intime','Levels_intime','perc_in_feature_intime']]
outtime=int_bivar[['keys_outtime','Levels_outtime','perc_in_feature_outtime']]

csi=pd.merge(intime,outtime,left_on=['keys_intime','Levels_intime'],right_on=['keys_outtime','Levels_outtime'],how='left')
csi['csi']=(csi['perc_in_feature_intime']-csi['perc_in_feature_outtime'])*np.log((csi['perc_in_feature_intime']/csi['perc_in_feature_outtime']))