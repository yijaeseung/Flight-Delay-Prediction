# Data Crawling using Selenium

from selenium import webdriver
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm_notebook
import pickle
import gc
import os


driver_path = r'C:\Users\이재승\Desktop\jason\weather'
loc_driver = driver_path + '\chromedriver'

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

driver = webdriver.Chrome(executable_path = loc_driver, chrome_options=options)


def get_df_ab(my_soup, date):
    data = {}
    comp, flight_num, dest, sche, real, my_type , status, reason = [], [], [], [] ,[] ,[] ,[], []
    for ind, each_row in enumerate(my_soup.find_all('tr')[1:-1] ):

        if ind % 2 ==0:
            vals =[]
            for ind , x in enumerate(each_row.find_all('td') ):
                if ind % 2 == 0:
                    vals.append(x.text.lstrip())

            comp.append(vals[0])
            flight_num.append(vals[1])
            dest.append(vals[2])
            sche.append(vals[3])
            real.append(vals[5])
            my_type.append(vals[6])
            status.append(vals[7])
            reason.append(vals[8])

    df = pd.DataFrame(data=[comp, flight_num, dest, sche, real, my_type, status, reason]).T
    df.columns = ['Company','Flight_Num','Dep','Sche','Dept','Type','Status', 'Reason']
    df['Date'] = date
    return df

def get_df_nor(my_soup, date):
    data = {}
    comp, flight_num, dest, sche, real, my_type , status, = [], [], [], [] ,[] ,[] ,[]
    for ind, each_row in enumerate(my_soup.find_all('tr')[1:-1] ):

        if ind % 2 ==0:
            vals =[]
            for ind , x in enumerate(each_row.find_all('td') ):
                if ind % 2 == 0:
                    vals.append(x.text.lstrip())

            comp.append(vals[0])
            flight_num.append(vals[1])
            dest.append(vals[2])
            sche.append(vals[3])
            real.append(vals[5])
            my_type.append(vals[6])
            status.append(vals[-1])

    df = pd.DataFrame(data=[comp, flight_num, dest, sche, real, my_type, status]).T
    df.columns = ['Company','Flight_Num','Dep','Sche','Dept','Type','Status']
    df['Date'] = date

    return df

def get_soup(url):   

    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
        
    return soup


def take_data(year, loc, status, dep_or_arr, path):
    my_dict = {}
    air_dict = {'incheon': 'RKSI', 'kimpo': 'RKSS', 'jeju': 'RKPC'}
    os.chdir(path + '\{}\{}\{}'.format(loc, dep_or_arr, status))
    if year == '2019':
        start_date = '2019-01-01'
        end_date = '2019-04-30'
    else:
        start_date = '{}-01-01'.format(year)
        end_date = '{}-12-31'.format(year)
        
    for date in tqdm_notebook(pd.date_range(start =  start_date, end= end_date) ):
        try:
            real_date = date.strftime('%Y%m%d')
            url_normal = 'http://www.airportal.go.kr/servlet/aips.life.airinfo.RbHanCTL?cmd=c_getList&depArr={}&current_date={}&airport={}&al_icao=&fp_id='.format(dep_or_arr,real_date,air_dict[loc])
            url_abnormal = 'http://www.airportal.go.kr/servlet/aips.life.airinfo.RbBejCTL?cmd=c_getList&airport={}&al_icao=&current_date={}&depArr={}&fp_id='.format(air_dict[loc], real_date, dep_or_arr)
            if status == 'normal': 
                soup = get_soup(url_normal)
                my_dict[real_date] = get_df_nor(soup, date)
            elif status == 'abnormal':
                soup = get_soup(url_abnormal)
                my_dict[real_date] = get_df_ab(soup, date)

        except:
            my_dict[real_date] = None
            gc.collect()
        
        
    for ind, key in tqdm_notebook( enumerate( my_dict.keys() )  ):
        if ind == 0 :
            data= my_dict[key]
        else:
            try:         
                data = pd.concat([data, my_dict[key]])
            except:
                print('{} failed'.format(year))
                continue
    data.to_csv('{}.csv'.format(year))
    
    return my_dict






path = r'C:\Users\이재승\Desktop\jason\weather\data'


years = [str(y) for y in range(2007, 2020)]

for year in years:
    take_data(year, 'jeju', 'abnormal', 'D', path)
    take_data(year, 'jeju', 'normal', 'D', path)
    print('{} finished'.format(year))
