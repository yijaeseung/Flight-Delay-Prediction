import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from itertools import product
import datetime
import seaborn as sns
import random
import lightgbm as lgb
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score
get_ipython().run_line_magic('matplotlib', 'notebook')


def display_all(df):
         with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
                display(df)
                
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'rmse_', rmse(labels, preds) , False


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz


def add_datepart(df, fldname, drop=True, time=False, errors="raise"):	
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df 
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.
    Parameters:
    -----------
    df: The data frame you wish to process.
    y_fld: The name of the response variable
    skip_flds: A list of fields that dropped from df.
    ignore_flds: A list of fields that are ignored during processing.
    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)
    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.
    preproc_fn: A function that gets applied to df.
    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.
    subset: Takes a random subset of size subset from df.
    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time (mean and standard deviation).
    Returns:
    --------
    [x, y, nas, mapper(optional)]:
        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.
        y: y is the response variable
        nas: returns a dictionary of which nas it created, and the associated median.
        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous
        variables which is then used for scaling of during test-time.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> x, y, nas = proc_df(df, 'col1')
    >>> x
       col2
    0     1
    1     2
    2     1
    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])
    >>>round(fit_transform!(mapper, copy(data)), 2)
    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


# 2. Data Exploration

os.chdir(path + '\merged')
jj = pd.read_csv('jeju.csv')


jj = jj.loc[jj['dep_loc'] == 'GMP(김포)']


# Calculating delay time to distinguish flights delayed for more than one hour

def cal_delay_time(row):
    if row['real_dep'] == ':':
        return np.nan
    else:
        scheduled_dep = int(row['scheduled_dep'].split(':')[0])*60 + int(row['scheduled_dep'].split(':')[1])
        real_dep  = int(row['real_dep'].split(':')[0])*60 + int(row['real_dep'].split(':')[1])
        delay_time = real_dep - scheduled_dep
        
        return delay_time


jj['delay_time'] = jj.apply(cal_delay_time, axis=1)


def convert_to_datetime(time):
    try:
        time_fixed = datetime.datetime.strptime(time,'%H:%M').time()
        return(time_fixed)
    except:
        return(time)

for col in ['real_dep', 'scheduled_dep']:
    jj[col] = jj[col].apply(convert_to_datetime)


# Just using major airlines for consistency of prediction

major_airlines = jj['airline'].value_counts()[:7].index
jj = jj.loc[jj['airline'].isin(major_airlines)]


jj['status'].value_counts()


jj = jj.loc[(jj['delay_time'] < 600)|(jj['status'].isin(['취소', '회항']))]
jj = jj.loc[(jj['delay_time'] > - 60)|(jj['status'].isin(['취소', '회항']))]


# 2-1. Check the Data

print(jj.shape)

display_all(jj.head())


jj.dtypes


# - object features: date, real_dep, scheduled_dep
# - numerical features: wind_direction, wind_speed, visible_dist, cloud_cover_1~3, ceiling_1~3, hourly_temp, dew_point, sea_level_atmo_pressure, atmo_pressure, hourly_precip, max_wind_speed_time, max_instant_wind_speed_time, hourly_max_precip_time, delay_time
# - categorical features: phenomena, airline, flight_num, reason, status, type, dep_loc
# - datetime features: time

# - Missing Data

def missing_data(data):
    total = data.isnull().sum()
    percent = round((data.isnull().sum() / data.isnull().count()) * 100, 2)
    tt = pd.concat([total, percent], axis=1, keys= ['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtypes)
        types.append(dtype)
    tt['Types'] = types
    return np.transpose(tt)


display_all(missing_data(jj))


# - Null Values Manipulation

# a. airline

jj.loc[jj['airline'].isnull(), 'flight_num'].value_counts().head(10)


# - I found that lots of observations without airline are private or military flights. I dropped those observations since making prediction for those flights isn't my goal.

jj = jj.loc[jj['airline'].notnull()]


# b. Status

jj['status'].unique()


len(jj.loc[(jj['status'].isnull())])


jj.loc[(jj['status'].isnull()), 'real_dep'].value_counts().head()


# - We can find that majority of flights without status also don't have the real departure time, even though they were not cancelled flights excluding just one flight. We can use those observations since our target to predict is the delay time, the gap between scheduled and real departure time. Hence, I dropped those observations.

jj = jj.loc[jj['status'].notnull()]


# c. type

jj.loc[jj['type'].isnull(), 'airline'].value_counts()


# - All flights without type value are for carrying passengers. Hence, I gave them passenger values.

jj.loc[jj['type'].isnull(), 'type'] = '여객'


# d. Weather features

jj.loc[jj['wind_direction_dep'].isnull(), 'time'].value_counts()


# - Weather data of this time is missing. I dropped those observations.

jj = jj.loc[jj['wind_direction_dep'].notnull()]


jj.loc[jj['wind_direction_des'].isnull(), 'time'].value_counts()


jj = jj.loc[jj['wind_direction_des'].notnull()]


# e. delay time

jj.loc[jj['delay_time'].isnull(), 'status'].value_counts()


# - All of observations which don't have delay time are cancellation (취소) or delay (지연), except for only one observation. I dropped that observation.

# jj = jj.drop([203369])


# - There can be some accidental delays that can't be covered by the data that I uses. Hence, I decided to exclude the observations with delay time > 120 or delay time < -30

# Manipulation Result

display_all(missing_data(jj))



# 2-2. Individual Features Visualization

# - object features: date, real_dep, scheduled_dep
# - numerical features: wind_direction, wind_speed, visible_dist, cloud_cover_1~3, ceiling_1~3, hourly_temp, dew_point, sea_level_atmo_pressure, atmo_pressure, hourly_precip, max_wind_speed_time, max_instant_wind_speed_time, hourly_max_precip_time, delay_time
# - categorical features: phenomena, airline, flight_num, reason, status, type, dep_loc
# - datetime features: time

sns.boxplot('delay_time', data=jj)


# - Delay Time

plt.figure()
plt.hist(jj['delay_time'], normed=True, bins=500)


# Weather Features

wea_num_cols = ['hourly_temp', 'dew_point', 'sea_level_atmo_pressure', 'atmo_pressure','hourly_precip']


wea_cat_cols = ['wind_direction','wind_speed', 'visible_dist', 'phenomena', 'cloud_cover_1','cloud_cover_2', 'cloud_cover_3', 'ceiling_1', 'ceiling_2', 'ceiling_3',
               'max_wind_speed_time', 'max_instant_wind_speed_time','hourly_max_precip_time']

for col in wea_cat_cols:
    fig, axes = plt.subplots(1, 2, figsize= (10, 5))
    plt.subplot(1, 2, 2)
    sns.boxplot(jj[col], jj['delay_time'])
    plt.xticks(rotation='vertical')
    plt.title(col)

    plt.subplot(1, 2, 1)
    plt.hist(jj[col], bins = jj[col].nunique())
    plt.xticks(rotation='vertical')
    plt.title(col)
    plt.show()


# We can find that:
# - Delay time becomes longer as wind speed becomes high.
# - Delay time becomes longer as visible distance short.
# - Delay time was longer for specific meteorological phenomena.
# - Delay time becomes longer when the cloud coverage for first cloud layer was the maximum value (9.0).


for col in wea_num_cols:
    plt.figure()
    sns.jointplot(jj[col], jj['delay_time'], joint_kws={'line_kws':{'color':'limegreen'}}, kind='reg')
    plt.show()


# It is hard to say that there are significant correlation between numeric weather features and delay time.

# - Categorical Features

airline_delay = jj.groupby(['airline'])['delay_time'].agg({'count': len, 'mean': np.mean, 'std': np.std})
airline_delay[airline_delay['count'] > 10000]


reason_delay = jj.groupby(['reason'])['delay_time'].agg({'count': len, 'mean': np.mean, 'std': np.std})
reason_delay.sort_values(by='count', ascending=False).head(20)


# We can see that visible distance and wind speed are important factors for delay caused by bad weather condition.

status_delay = jj.groupby(['status'])['delay_time'].agg({'count': len, 'mean': np.mean, 'std': np.std})
status_delay


type_delay = jj.groupby(['type'])['delay_time'].agg({'count': len, 'mean': np.mean, 'std': np.std})
type_delay


# There are only five cargo planes. I decided to drop them 

jj = jj.loc[jj['type'] != '화물']


# - Datetime Features

# Will be updated



