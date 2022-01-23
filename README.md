
#  Bivariates and Character Stability Index (CSI)

This script gives a function to easily calculate Bivariates and the character Stability Index for the variables in the data. This can be used only when the Y/target variable is binary i.e 0/1. 

The metrics being calculated are **Weight of  Evidence** and **Information Value**. These metrics help us identify important variables with respect to our binary Y variable
## Usage  Scenarios
1. For feature selection to select the variables based on WOE and IV on train data
2. During Model monitoring where we can check whether the new data has the same distribution and information as model trained data

    Check the given [Ipython Notebook](https://github.com/vasanth3029/Bivariates-and-Character-Stability-Index/blob/master/CSI%20after%20bivar.ipynb) for detailed steps on how to use the function and interpret the results.
## Key Features

- Uses different methods for object/categorical and Numerical columns
- Binning is done automatically for continuous variables
- Bins are calculated based on train data and the same bins are applied to out-of-time data to make
    the bins common. This allows us to check if some bins have changed drastically in our new data
- Null values are imputed and in numerical columns, they are kept in a separate bin so as not to affect the other bins.
- The output for train data and out of time data contains the same levels and bins for easy comparison(Except if we have new values in categorical variables for OOT data)
- Straight forward CSI calculation with the help of bivariates output
## Important links

- [WOE and IV](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Data used in the notebook](https://www.kaggle.com/sonujha090/bank-marketing)
