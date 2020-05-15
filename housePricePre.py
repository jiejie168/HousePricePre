__author__ = 'Jie'
"""
House Prices: Advanced Regression Techniques
Prdict the sales price, and practice feature engineering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.pipeline import make_pipeline


class DataVisul():

    def dataShape(self,train_df,test_df,all_df):
        print ("the shape of train_data is:{}".format(train_df.shape))
        print ("the shape of test_data is:{}".format(test_df.shape))
        print ("the shape of all_data is:{}".format(all_df.shape))

    def dataShow(self,df):
        print ("the first 5 rows of data\n")
        print (df.head())
        print ("the last 5 rows of data\n")
        print (df.tail())
        print ("the non-null type and data type\n")
        print (df.info())
        print ("the statistic infor of data\n")
        print (df.describe().T)

    def dataPlot(self,df,target):
        # check the target data (a series) and its skewness and kurtosis
        plt.figure(figsize=(8,6))
        # fig,((ax1,ax2))=plt.subplots(1,2)
        # plot the variant distribution
        sns.distplot(df[target],kde_kws={'color':'Navy'})
        plt.figure(figsize=(8,6))
        sns.boxplot(df[target])
        print ("the skewness (unsymmetry) of data is: {}".format(df[target].skew()))
        print ("the curtosis of data is: {}".format(df[target].kurt()))
        print ("check the skewed distribution of data to see if a transformation is needed")
        plt.show()

    def featureSelectView(self,df,target):
        # obtain the correlation coefficients
        # select the independent variables with high correlation coeff, w.r.t. target
        plt.figure(figsize=(16,10))
        cor=df.corr()  # obtain the correlation coefficient of each variant
        sns.heatmap(cor,annot=True,cmap=plt.cm.Reds,vmax=0.9)

        # cor_target=abs(cor[target].values) # transform series to array
        cor_target=abs(cor[target]) # no need to transform to array.
        cor_boolen=cor[cor_target>=0.5]  # obtain the df with boolen mask. only the rows satisfying this condition remains
        relev_features=cor_boolen[target].sort_values(ascending=False)[1:]
        print ("the possible significant features are : {}".format(relev_features))
        # plt.show()
        return cor

    def databoxPlotVars(self,df,var,target):
        # plot figures btw different variables
        plt.figure(figsize=(12,6))
        sns.boxplot(df[var],df[target])
        plt.show()

    def dataScatterPlotVars(self,df,var,target):
        # plot figures btw different variables
        plt.figure(figsize=(12,8))
        sns.scatterplot(df[var],df[target])
        plt.show()

class DataClean():

    def dataLogTransform(self,df,target):
        # if the target data are skewed, it needs to do a Log(e) transformation
        # so that the data are close to a normal distribution
        plt.figure(figsize=(8,6))
        df[target]=np.log(df[target]+1)
        sns.distplot(df[target])
        # plt.show()
        return df[target]

    def removeExcept(self,df,var,target):
        # if there are large outliers or exceptional data for specific columns, remove them
        # such operation could be useful for the fitted models.
        # here only use the var: 'GrLivArea', and target: 'SalePrice' as an example
        # df[target]=np.log(df[target]+1)
        df[target]=self.dataLogTransform(df,'SalePrice')
        index_drop=df[(df[var]>4000) & (df[target]<12.5)].index # boolen mask
        df=df.drop(index_drop)
        df.reset_index(drop=True, inplace=True)
        return df

    def check_missData(self,df):
        # check the missing data of dataFrame.
        # output them in the form of dataFrame
        # sum(): count the number of missing data in each column
        # isnull(): return a dataFrame with True, and False
        miss_tot=df.isnull().sum().sort_values(ascending=False)
        counts_all=df.isnull().count() # count all the elements, including the missing elements
        miss_per=((df.isnull().sum())*100/counts_all).sort_values(ascending=False)
        miss_all=pd.concat([miss_tot,miss_per],axis=1,keys=['TotalNum','TotalPerc'])
        return miss_all


    def check_valsCount(self,df):
        cols=df.columns
        for col in cols:
            counts=df[col].value_counts()
            print (counts)
            print ('\n')

    def dataSplit(self,df):
        # split data by their types: float64,int64,object
        df_cat=df.select_dtypes('object')
        df_num=df.select_dtypes(['int64','float64'])
        df_cat_null=df_cat.isnull().sum().sort_values(ascending=False)
        df_num_null=df_num.isnull().sum().sort_values(ascending=False)
        return df_cat,df_num,df_cat_null,df_num_null

    def fillnan(self,df,columns,fills):
        # fill the missing data in df with 'fills'
        for col in columns:
            df[col].fillna(fills,inplace=True)
        return df

    def fillnan_1(self,df,columns):
        # fill the missing data in df with 'fills'. fills vary with col.
        # mode(): statistically, it is the X value corresponding to the largest PDF.
        for col in columns:
            df[col].fillna(df[col].mode()[0],inplace=True)
        return df

    def num2str(self,df,var):
        df[var]=df[var].apply(str)

    def skew_check(self,df):
        df_num=df.select_dtypes(['float64','int64'])
        skews_df=df_num.apply(lambda x:x.skew()).sort_values(ascending=False)
        return skews_df

    def skewed_transform(self,df):
        # transform the skewed,non-normal distribution data to
        # the normal distribution data
        skews_df=self.skew_check(df) # the skewed data in df
        high_skews=skews_df[abs(skews_df)>0.5] # select the high_skews larger than 0.5
        # transform the skewed data; why +1, is not clear
        for ind in skews_df.index:
            df[ind]=boxcox1p(df[ind],boxcox_normmax(df[ind]+1))
        return df

class Model_fit():

        def features(self,df,droped):
            # get the final features from the cleaned data
            df.drop([droped],axis=1,inplace=True)
            features_df=pd.get_dummies(df) # one-hot-encode
            features_df.reset_index(drop=True) # drop=True means refresh the index in order
            print ("the final shapes of all features: {}".format(features_df.shape))
            return features_df

        def train_test_split(self,nrows,features_df):
            # split the overall data into train set and test set
            # nrows=train_df.shape[0]
            X_train=features_df[:nrows]
            X_test=features_df[nrows:]
            return X_train,X_test

        def modelfit(self,X_train,y_train):
            # fit the models, by grid search
            alphas_1=[14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
            alphas_2=[5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
            alphas_3=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
            e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1] # it should between [0,1]

            kfold=KFold(n_splits=10,shuffle=False,random_state=42)
            ## use three different ML regressors for fitting
            ## first construct the regressor
            lineRidge=make_pipeline(RobustScaler(),RidgeCV(alphas=alphas_1,cv=kfold))
            lineLasso=make_pipeline(RobustScaler(),LassoCV(alphas=alphas_2,max_iter=1e7,cv=kfold,random_state=42))
            lineElastic=make_pipeline(RobustScaler(),ElasticNetCV(alphas=alphas_3,max_iter=1e7,
                                                                  cv=kfold,l1_ratio=e_l1ratio,random_state=42))
            ### fit the model
            lineRidge.fit(X_train,y_train)
            lineLasso.fit(X_train,y_train)
            lineElastic.fit(X_train,y_train)
            return lineRidge,lineLasso,lineElastic

        def blending(self,X_train,y_train,X_test):
            # bending the prediction models from multiple regressors
            lineRidge,lineLasso,lineElastic=self.modelfit(X_train,y_train)
            return ((lineRidge.predict(X_test))+(lineLasso.predict(X_test))+(lineElastic.predict(X_test)))/3

        def adjust_predict(self,results):
            # Fix outleir predictions, this is done by taking reference from other people.
            q1 = results['SalePrice'].quantile(0.0045) # Return value at the given quantile
            q2 = results['SalePrice'].quantile(0.99)
            results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
            results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

        def resultsOut(self,X_train,y_train,X_test):
            # An array with exponential(all elements of input array) - 1.
            results=np.expm1(self.blending(X_train,y_train,X_test))
            results=pd.DataFrame({'SalePrice':results})
            return results

def main():
    ################################
    ## read data from csv files
    train_data=pd.read_csv("train.csv")
    test_data=pd.read_csv("test.csv")
    # clean,and  manipulate data together.
    dataAll=pd.concat([train_data,test_data],ignore_index=True)

    ###########################################
    ## 1st round of data visulization, and some cleaning
    dataVisul=DataVisul()
    dataClean=DataClean()
    dataVisul.dataShape(train_data,test_data,dataAll)
    # dataVisul.dataShow(dataAll)
    # dataVisul.dataPlot(train_data,'SalePrice')
    # cor=dataVisul.featureSelectView(train_data,'SalePrice')
    train_data=dataClean.removeExcept(train_data,'GrLivArea','SalePrice')
    # dataVisul.dataPlotVars(train_data,'OverallQual','SalePrice')
    # dataVisul.dataScatterPlotVars(train_data,'GrLivArea','SalePrice')

    ########################################################
    ## 2nd round of data combination after the first checking
    dataAll=pd.concat([train_data,test_data])
    dataAll.reset_index(drop=True)
    # dataClean.check_valsCount(dataAll)
    # miss_df=dataClean.check_missData(dataAll)
    df_cat,df_num,df_cat_null,df_num_null=dataClean.dataSplit(dataAll)
    # print(df_cat_null)
    # print (df_num_null)

    ##############################################
    ## after checking the missing data.
    ## filling the data in a proper way
    # There are two ways to fill the missing values in the categorical values.
    # 1) fill with 'None'
    #    e.g. Alley: Type of alley access to property
    #         Grvl	Gravel
    #         Pave	Paved
    #         NA 	No alley access.   it means: no access,hence replace it with sign'None'
    # fill null values by 'None'
    columns1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',
           'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure',
           'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
    #2) for the others, fill the Nan with mode of each column
    columns2 = ['MSZoning','Functional', 'Utilities', 'Electrical', 'KitchenQual', 'SaleType',
           'Exterior2nd', 'Exterior1st']
    # select the feature with Nan, as show above.
    columns3 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFullBath',
       'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'GarageCars',
       'BsmtFinSF2', 'BsmtFinSF1']
    df_cat_1=dataClean.fillnan(df_cat,columns1,'None')
    df_cat=dataClean.fillnan_1(df_cat_1,columns2)
    df_num=dataClean.fillnan(df_num,columns3,0)
    # miss_df=dataClean.check_missData(df_cat)
    miss_df=dataClean.check_missData(df_cat)
    # print (df_cat.shape)
    # print (df_num.shape)

    #####################################################
    ###### combine data again
    df=pd.concat([df_cat,df_num],axis=1)
    # print(df.isnull().sum().sort_values(ascending=False))
    dataClean.num2str(df,'MSSubClass')
    dataClean.num2str(df,'YrSold')
    dataClean.num2str(df,'MoSold')
    df.drop('SalePrice',axis=1,inplace=True)## this column is the predict target, so it can be removed here.
    # miss_df=dataClean.check_missData(df)
    # print (miss_df)
    skews_df=dataClean.skew_check(df)
    ## since there are many skewed data, we have to do another transformation.
    # dataVisul.dataPlot(df,'3SsnPorch')
    df=dataClean.skewed_transform(df)  # df is the overall dataFrame including skewed, and no-skewed
    print (df.shape)
    # dataVisul.dataPlot(df,'3SsnPorch')
    # print (df['Id'])

    #########################################
    ## models fitting.
    model_fit=Model_fit()
    features_df=model_fit.features(df,'Id')

    y_train=train_data['SalePrice']
    nrows=train_data.shape[0]
    X_train,X_test=model_fit.train_test_split(nrows,features_df)
    lineRidge,lineLasso,lineElastic=model_fit.modelfit(X_train,y_train)
    final_pred=model_fit.blending(X_train,y_train,X_test) # these results should transform back.
    results=model_fit.resultsOut(X_train,y_train,X_test)
    model_fit.adjust_predict(results)

    ## save models
    ### in case for future use, there is no need to refit, just load the model and do predictions
    from sklearn.externals import joblib
    joblib.dump(lineRidge,"D:/python-ml/kaggles_competition/HousePricePre/lineRidge.pkl")
    joblib.dump(lineLasso,"D:/python-ml/kaggles_competition/HousePricePre/lineLasso.pkl")
    joblib.dump(lineElastic,"D:/python-ml/kaggles_competition/HousePricePre/lineElastic.pkl")

    # clf1=joblib.load("D:/python-ml/kaggles_competition/HousePricePre/lineRidge.pkl")
    # clf2=joblib.load("D:/python-ml/kaggles_competition/HousePricePre/lineLasso.pkl")
    # blending clf1,clf2, and prediction

    ### final results arrangement
    results['Id']=test_data['Id']
    results.to_csv("D:/python-ml/kaggles_competition/HousePricePre/housePricePre_Jason.csv",
                   index=False)
    print (results.shape)
    print ("Success !")

if __name__ == '__main__':
    main()
