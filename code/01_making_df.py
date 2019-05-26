# 1. Loading Functions and Making Dataset

# 1-1. Loading Functions

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


# 1-2. Making Dataset

path = r'C:\Users\이재승\Desktop\jason\weather\data'


def make_dataset(year, loc, path):
    year = str(year)
    
    
    # load_dataset
    os.chdir(path + '\{}\D\normal'.format(loc))
    normal_df = pd.read_csv('{}.csv'.format(year))
    
    os.chdir(path + '\{}\D\abnormal'.format(loc))
    abnormal_df = pd.read_csv('{}.csv'.format(year))

    
    if year == '2019':
        encoding = 'utf-8'
    else:
        encoding = 'cp949'
        
#     os.chdir(r'C:\Users\이재승\Desktop\jason\weather\data\weather\minutely')
#     min_df = pd.read_csv('{}_{}.csv'.format(loc, year), encoding=encoding)
    
    os.chdir(path + '\weather\hourly')
    hour_dep = pd.read_csv('{}_{}.csv'.format(loc, year), encoding=encoding)
    
    if loc == 'jeju':
        hour_des = pd.read_csv('kimpo_{}.csv'.format(year), encoding=encoding)

    if year in ['2018', '2019']:
        hour_dep = hour_dep[['지점', '일시', '풍향(deg)', '풍속(KT)', '순간풍속(KT)', '시정(m)', '일기현상',
       '1층 운량(1/8)', '2층 운량(1/8)', '3층 운량(1/8)', '4층 운량(1/8)',
       '1층 운형', '2층 운형', '3층 운형', '4층 운형', '1층 운고(FT)', '2층 운고(FT)',
       '3층 운고(FT)', '4층 운고(FT)', '기온(°C)', '이슬점온도(°C)', '해면기압(hPa)',
       '현지기압(hPa)', '강수량(mm)']]
        hour_des = hour_des[['지점', '일시', '풍향(deg)', '풍속(KT)', '순간풍속(KT)', '시정(m)', '일기현상',
       '1층 운량(1/8)', '2층 운량(1/8)', '3층 운량(1/8)', '4층 운량(1/8)',
       '1층 운형', '2층 운형', '3층 운형', '4층 운형', '1층 운고(FT)', '2층 운고(FT)',
       '3층 운고(FT)', '4층 운고(FT)', '기온(°C)', '이슬점온도(°C)', '해면기압(hPa)',
       '현지기압(hPa)', '강수량(mm)']]
    
    else:        
        hour_dep = hour_dep[['지점', '일시', '풍향(deg)', '풍속(KT)', '순간풍속(KT)', '시정(m)', '일기현상(null)',
           '1층 운량(1/8)', '2층 운량(1/8)', '3층 운량(1/8)', '4층 운량(1/8)', '1층 운형(null)',
           '2층 운형(null)', '3층 운형(null)', '4층 운형(null)', '1층 운고(FT)', '2층 운고(FT)',
           '3층 운고(FT)', '4층 운고(FT)', '기온(°C)', '이슬점온도(°C)', '해면기압(hPa)',
           '현지기압(hPa)', '강수량(mm)']]
        hour_des = hour_des[['지점', '일시', '풍향(deg)', '풍속(KT)', '순간풍속(KT)', '시정(m)', '일기현상(null)',
           '1층 운량(1/8)', '2층 운량(1/8)', '3층 운량(1/8)', '4층 운량(1/8)', '1층 운형(null)',
           '2층 운형(null)', '3층 운형(null)', '4층 운형(null)', '1층 운고(FT)', '2층 운고(FT)',
           '3층 운고(FT)', '4층 운고(FT)', '기온(°C)', '이슬점온도(°C)', '해면기압(hPa)',
           '현지기압(hPa)', '강수량(mm)']]

    os.chdir(path + '\weather\daily')
    day_df = pd.read_csv('{}_{}.csv'.format(loc, year), encoding=encoding)
    day_df = day_df[['지점', '일시', '최대풍속 나타난시각(hhmi)', '최대순간풍속 나타난시각(hhmi)', '최고기온(°C)',
       '최저기온(°C)', '최고기온시각(hhmi)', '최저기온시각(hhmi)', '평균해면기압(hPa)',
       '최고해면기압(hPa)', '최저해면기압(hPa)', '강수량합(mm)', '1시간최다강수(mm)',
       '1시간최다강수 시각(hhmi)', '30분최다강수(mm)', '10분최다강수(mm)', '최심신적설(cm)',
       '최심적설(cm)', '평균이슬점온도(°C)', '평균상대습도(%)']]
    
    # Flight dataset
    normal_df['Reason'] = '정상 운행'
    df = pd.concat([normal_df, abnormal_df], ignore_index=True)
    df.drop(['Unnamed: 0'], 1, inplace=True)

    df.columns = ['airline', 'date', 'dep_loc', 'real_dep', 'flight_num', 'reason', 'scheduled_dep', 'status', 'type']

    df = df[(df.scheduled_dep != ':')&(df.airline != '항공사')]
    df['time'] = df['date'] + ' ' + df['scheduled_dep'].map(lambda x: x.split(':')[0] + x.split(':')[1])
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

#     # Merging Minutely Data
#     min_df.columns = ['loc', 'time', 'temp', '10min_avg_wind_speed', '10min_avg_wind_dir', '10min_avg_MOR', '10min_avg_RVR', 'cum_precip']

#     min_df['time'] = pd.to_datetime(min_df['time'])
#     flight_time = df.time.values
#     min_data = min_df.loc[min_df['time'].isin(flight_time)]
#     df = df.merge(min_data, how='left', on=['time'])
#     df.drop(['loc'], 1, inplace=True)
    
    # Merging Hourly Data
    def add_time_for_hour_dep(time, delta):
        if str(time)[-5:] == '00:00':
            return time
        else:
            try:
                revised_time = pd.to_datetime(str(time[:13]) + ':00:00') + timedelta(hours=delta)
                return revised_time
            except:
                return np.nan  

    for i, hour_df in enumerate([hour_dep, hour_des]):
        if year in ['2018','2019']:
            hour_df.drop(['1층 운형','2층 운형', '3층 운형', '4층 운형'], 1, inplace=True)
        else:
            hour_df.drop(['1층 운형(null)','2층 운형(null)', '3층 운형(null)', '4층 운형(null)'], 1, inplace=True)
        hour_col_name = ['loc', 'time', 'wind_direction', 'wind_speed', 'instant_wind_speed', 'visible_dist', 'phenomena',
                            'cloud_cover_1', 'cloud_cover_2', 'cloud_cover_3', 'cloud_cover_4', 'ceiling_1', 'ceiling_2', 'ceiling_3', 'ceiling_4',
                            'hourly_temp', 'dew_point', 'sea_level_atmo_pressure', 'atmo_pressure', 'hourly_precip']
        hour_df.columns = hour_col_name


        to_drop = ['instant_wind_speed', 'cloud_cover_4', 'ceiling_4']
        hour_df.drop(to_drop, 1, inplace=True)
        cloud_cols = ['cloud_cover_1', 'cloud_cover_2', 'cloud_cover_3']
        hour_df[cloud_cols] = hour_df[cloud_cols].fillna(0)

        ceiling_cols = ['ceiling_1', 'ceiling_2', 'ceiling_3']
        hour_df[ceiling_cols] = hour_df[ceiling_cols].fillna(0)

        hour_df.loc[hour_df['hourly_precip'] == 0, 'hourly_precip'] = 0.01
        hour_df['hourly_precip'] = hour_df['hourly_precip'].fillna(0)

        hour_df['time'] = pd.to_datetime(hour_df['time'], errors='coerce')

        df['time_for_hourly_data'] = df['time'].apply(lambda x: add_time_for_hour_dep(str(x), i + 1))

        if i == 0:
            hour_df.columns = [col + '_dep' for col in list(hour_df)]
            df = df.merge(hour_df, how='left', left_on = 'time_for_hourly_data', right_on = 'time_dep').drop(['time_dep', 'time_for_hourly_data', 'loc_dep'], 1)
        elif i == 1:
            hour_df.columns = [col + '_des' for col in list(hour_df)]        
            df = df.merge(hour_df, how='left', left_on = 'time_for_hourly_data', right_on = 'time_des').drop(['time_des', 'time_for_hourly_data', 'loc_des'], 1)

    
    # Merging Daily Data
    day_df.columns = ['loc', 'date', 'max_wind_speed_time', 'max_instant_wind_speed_time', 'max_temp', 'min_temp',
                   'max_temp_time', 'min_temp_time', 'avg_sea_level_pressure', 'max_sea_level_pressure', 'min_sea_level_pressure',
                   'sum_precip', 'hourly_max_precip', 'hourly_max_precip_time', '30min_max_precip', '10min_max_precip', 'max_fresh_snow_cover',
                   'max_snow_cover', 'avg_dew_point', 'avg_relative_humidity']

    time_cols = [c for c in list(day_df) if 'time' in c]

    day_df[time_cols] = day_df[time_cols].applymap(lambda x: str(x))

    day_df.loc[day_df['hourly_max_precip_time'].notnull(), 'hourly_max_precip_time'] = day_df.loc[day_df['hourly_max_precip_time'].notnull(), 'hourly_max_precip_time'].apply(lambda x: x.split('.')[0])

    day_df[time_cols] = day_df[time_cols].applymap(lambda x: x.zfill(4))

    for col in time_cols:
        day_df.loc[day_df[col] == '0nan', col] = np.nan
        day_df.loc[day_df[col].notnull(), col] = day_df.loc[day_df[col].notnull(), 'date'] + ' ' + day_df.loc[day_df[col].notnull(), col].apply(lambda x: x[:2] + ':' + x[2:])
        day_df[col] = pd.to_datetime(day_df[col], errors = 'coerce')

    cols = ['max_wind_speed_time', 'max_instant_wind_speed_time', 'hourly_max_precip_time']
    for i, row in day_df.iterrows():
        for col in cols:
            try:
                time_range = pd.date_range(start= row[col] - timedelta(minutes= 30) , end = row[col] + timedelta(minutes=30), freq='1min')
                df.loc[df['time'].isin(time_range), col] = 1
            except:
                continue

    df[cols] = df[cols].fillna(0)
    
    return df
    


years = range(2007, 2020)

jj = []

for year in years:    
    temp = make_dataset(year, 'jeju', path)
    jj.append(temp)
    print('jeju_{}_finished'.format(year))

jj_df = pd.concat(jj)


jj_df.head()


os.chdir(path + '\merged')
jj_df.to_csv('jeju.csv', index=False)


