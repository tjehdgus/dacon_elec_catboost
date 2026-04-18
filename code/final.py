import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import random as rn
from datetime import datetime, timedelta
import warnings
import os
import math
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

# 한국어 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def summer_cos(date):
    """여름 계절 패턴 - 코사인 인코딩"""
    start_date = datetime.strptime("2024-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime("2024-09-14 00:00:00", "%Y-%m-%d %H:%M:%S")
    period = (end_date - start_date).total_seconds()
    return math.cos(2 * math.pi * (date - start_date).total_seconds() / period)

def summer_sin(date):
    """여름 계절 패턴 - 사인 인코딩"""
    start_date = datetime.strptime("2024-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime("2024-09-14 00:00:00", "%Y-%m-%d %H:%M:%S")
    period = (end_date - start_date).total_seconds()
    return math.sin(2 * math.pi * (date - start_date).total_seconds() / period)

def week_of_month(date):
    """월 내 주차 패턴 - 격주 일요일 패턴"""
    first_day = date.replace(day=1)
    if (date.isocalendar().week - first_day.isocalendar().week + 1) % 2 == 0:
        if date.weekday() == 6:  # 일요일 = 6
            return 1
    return 0

def is_building_holiday_updated(row):
    """백화점 건물별 휴무 패턴"""
    building_num = row['building_number']
    date = row['date_time']
    day_of_week = row['day_of_week']
    day = row['day']
    month = row['month']
    
    # 백화점 건물별 특별 휴무 패턴
    if building_num == 18:
        return day_of_week == 6
        
    elif building_num == 19:
        specific_holidays = ['2024-06-10', '2024-07-08', '2024-08-19']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 27:
        specific_holidays = ['2024-06-09', '2024-06-23', '2024-07-14', '2024-07-28', '2024-08-11', '2024-08-25']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 29:
        if day == 10:
            return True
        specific_holidays = ['2024-06-23', '2024-07-28', '2024-08-25']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 32:
        specific_holidays = ['2024-06-10', '2024-06-24', '2024-07-08', '2024-07-22', '2024-08-12', '2024-08-26']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 34:
        return False
        
    elif building_num == 40:
        specific_holidays = ['2024-06-09', '2024-06-23', '2024-07-14', '2024-07-28', '2024-08-11', '2024-08-25']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 45:
        specific_holidays = ['2024-06-10', '2024-07-08', '2024-08-19']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 54:
        specific_holidays = ['2024-06-17', '2024-07-01', '2024-08-19']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 59:
        specific_holidays = ['2024-06-09', '2024-06-23', '2024-07-14', '2024-07-28', '2024-08-11', '2024-08-25']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 63:
        specific_holidays = ['2024-06-09', '2024-06-23', '2024-07-14', '2024-07-28', '2024-08-11', '2024-08-25']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 73:
        return False
        
    elif building_num == 74:
        specific_holidays = ['2024-06-09', '2024-06-23', '2024-07-14', '2024-07-28', '2024-08-11', '2024-08-25',
                           '2024-06-17', '2024-07-01', '2024-08-26']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 79:
        specific_holidays = ['2024-06-17', '2024-07-01', '2024-08-19']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    elif building_num == 88:
        return False
        
    elif building_num == 95:
        specific_holidays = ['2024-07-08', '2024-08-05']
        return date.strftime('%Y-%m-%d') in specific_holidays
        
    else:
        # 일반 건물들: 주말 + 공휴일 적용
        official_holidays = ['2024-06-06', '2024-06-07', '2024-08-15', '2024-08-16']
        return (day_of_week >= 5) or (date.strftime('%Y-%m-%d') in official_holidays)

def enhanced_pattern_based_interpolation(building_data, target_indices, interpolation_type='zero'):
    """향상된 패턴 기반 스플라인 보간"""
    interpolated_values = []
    
    for idx in target_indices:
        target_datetime = building_data.loc[idx, 'date_time']
        target_hour = target_datetime.hour
        target_dow = target_datetime.dayofweek
        target_holiday = building_data.loc[idx, 'holiday'] if 'holiday' in building_data.columns else False
        
        # 보간 타입에 따른 패턴 매칭 전략
        if interpolation_type == 'zero':
            pattern_mask = (
                (building_data['date_time'].dt.hour == target_hour) & 
                (building_data['date_time'].dt.dayofweek == target_dow) &
                (building_data['power_consumption'] > 0) &
                (building_data.index != idx)
            )
            
            if 'holiday' in building_data.columns:
                pattern_mask = pattern_mask & (building_data['holiday'] == target_holiday)
            
        elif interpolation_type == 'outlier':
            if target_holiday:
                pattern_mask = (
                    (building_data['date_time'].dt.hour == target_hour) & 
                    (building_data['date_time'].dt.dayofweek == target_dow) &
                    (building_data['holiday'] == True) &
                    (building_data['power_consumption'] > 0) &
                    (building_data.index != idx)
                )
            else:
                pattern_mask = (
                    (building_data['date_time'].dt.hour == target_hour) & 
                    (building_data['holiday'] == True) &
                    (building_data['power_consumption'] > 0) &
                    (building_data.index != idx)
                )
        
        pattern_data = building_data[pattern_mask].copy()
        pattern_data = pattern_data.sort_values('date_time')
        
        if len(pattern_data) >= 4:
            try:
                base_date = building_data['date_time'].min()
                target_week = (target_datetime - base_date).days / 7.0
                
                pattern_weeks = []
                pattern_values = []
                
                for _, row in pattern_data.iterrows():
                    week_num = (row['date_time'] - base_date).days / 7.0
                    pattern_weeks.append(week_num)
                    pattern_values.append(row['power_consumption'])
                
                cs = CubicSpline(pattern_weeks, pattern_values, bc_type='natural')
                interpolated_value = cs(target_week)
                interpolated_value = max(0, interpolated_value)
                
                interpolated_values.append(interpolated_value)
                
            except Exception as e:
                fallback_value = pattern_data['power_consumption'].median()
                interpolated_values.append(fallback_value)
        
        else:
            interpolated_value = enhanced_fallback_interpolation(
                building_data, idx, target_hour, target_dow, target_holiday, interpolation_type
            )
            interpolated_values.append(interpolated_value)
    
    return np.array(interpolated_values)

def enhanced_fallback_interpolation(building_data, idx, target_hour, target_dow, target_holiday, interpolation_type):
    """향상된 패턴 데이터 부족 시 대체 보간 방법"""
    
    if interpolation_type == 'zero':
        same_hour_mask = (
            (building_data['date_time'].dt.hour == target_hour) & 
            (building_data['power_consumption'] > 0) &
            (building_data.index != idx)
        )
        
        if 'holiday' in building_data.columns:
            same_hour_mask = same_hour_mask & (building_data['holiday'] == target_holiday)
        
        same_hour_data = building_data[same_hour_mask]['power_consumption']
        
        if len(same_hour_data) >= 3:
            return same_hour_data.median()
    
    elif interpolation_type == 'outlier':
        if target_holiday:
            holiday_hour_mask = (
                (building_data['date_time'].dt.hour == target_hour) & 
                (building_data['holiday'] == True) &
                (building_data['power_consumption'] > 0) &
                (building_data.index != idx)
            )
        else:
            holiday_hour_mask = (
                (building_data['date_time'].dt.hour == target_hour) & 
                (building_data['holiday'] == False) &
                (building_data['power_consumption'] > 0) &
                (building_data.index != idx)
            )
        
        holiday_hour_data = building_data[holiday_hour_mask]['power_consumption']
        
        if len(holiday_hour_data) >= 3:
            return holiday_hour_data.median()
    
    # 추가 대체 방법들
    same_dow_mask = (
        (building_data['date_time'].dt.dayofweek == target_dow) & 
        (building_data['power_consumption'] > 0) &
        (building_data.index != idx)
    )
    
    if 'holiday' in building_data.columns:
        same_dow_mask = same_dow_mask & (building_data['holiday'] == target_holiday)
    
    same_dow_data = building_data[same_dow_mask]['power_consumption']
    
    if len(same_dow_data) >= 3:
        return same_dow_data.median()
    
    # 최후 방법
    all_nonzero = building_data[building_data['power_consumption'] > 0]['power_consumption']
    if len(all_nonzero) > 0:
        return all_nonzero.median()
    
    return 100.0

def apply_enhanced_spline_interpolation(df):
    """향상된 스플라인 보간 적용"""
    print("향상된 스플라인 보간 적용 중...")
    
    if df['date_time'].dtype == 'object':
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H')
    
    if 'hour' not in df.columns:
        df['hour'] = df['date_time'].dt.hour
    if 'day' not in df.columns:
        df['day'] = df['date_time'].dt.day
    if 'month' not in df.columns:
        df['month'] = df['date_time'].dt.month
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date_time'].dt.dayofweek
    
    if 'holiday' not in df.columns:
        df['holiday'] = df.apply(is_building_holiday_updated, axis=1).astype(int)
    
    # 특정 이상치 정의
    SPECIFIC_OUTLIERS = {
        30: [('2024-07-13', 20), ('2024-07-25', 0)],
        43: [('2024-06-10', 17), ('2024-06-10', 18), ('2024-08-12', 16), ('2024-08-12', 17)],
        52: [('2024-08-10', 0), ('2024-08-10', 2)],
        67: [('2024-06-10', 17), ('2024-06-10', 18), ('2024-08-12', 16), ('2024-08-12', 17)],
        81: [('2024-07-17', 14)],
        72: [('2024-07-21', 11)],
        80: [('2024-07-06', 9), ('2024-07-06', 10), ('2024-07-06', 11), ('2024-07-06', 12), 
             ('2024-07-06', 13), ('2024-07-06', 14), ('2024-07-06', 15),
             ('2024-07-08', 12), ('2024-07-08', 13),
             ('2024-07-20', 9), ('2024-07-20', 10), ('2024-07-20', 11), ('2024-07-20', 12), 
             ('2024-07-20', 13)],
        92: [('2024-07-17', 18), ('2024-07-17', 19), ('2024-07-17', 21)],
        29: [('2024-06-27', 1)],
        40: [('2024-07-14', 0)],
        79: [('2024-08-19', 3), ('2024-08-19', 4), ('2024-08-19', 5)],
        88: [('2024-08-23', 6), ('2024-08-23', 8)],
        95: [('2024-08-05', 11)],
        42: [('2024-07-17', 14)],
        44: [('2024-06-06', 13), ('2024-06-30', 0), ('2024-06-30', 2)],
        90: [('2024-06-05', 18)],
        41: [('2024-06-22', 1), ('2024-06-22', 4), ('2024-07-17', 14), ('2024-07-17', 15)],
        76: [('2024-06-03', 13), ('2024-06-20', 12), ('2024-06-20', 16)],
        70: [('2024-06-05', 9)],
        53: [('2024-06-15', 8), ('2024-06-15', 11)],
        94: [('2024-07-27', 9), ('2024-07-27', 12)],
        5: [('2024-08-04', 7), ('2024-08-04', 8)],
        8: [('2024-07-21', 8)],
        98: [('2024-06-13', 14)],
        68: [('2024-06-28', 23), ('2024-06-29', 1)]
    }
    
    df_interpolated = df.copy()
    
    for building_num in sorted(df_interpolated['building_number'].unique()):
        building_data = df_interpolated[df_interpolated['building_number'] == building_num].copy()
        
        if len(building_data) == 0:
            continue
        
        building_data = building_data.sort_values('date_time').reset_index(drop=True)
        
        # 특정 이상치 보간
        if building_num in SPECIFIC_OUTLIERS:
            outlier_points = SPECIFIC_OUTLIERS[building_num]
            outlier_indices = []
            
            for date_str, hour in outlier_points:
                target_datetime = pd.to_datetime(f"{date_str} {hour:02d}:00:00")
                mask = building_data['date_time'] == target_datetime
                if mask.any():
                    idx = building_data[mask].index[0]
                    outlier_indices.append(idx)
            
            if len(outlier_indices) > 0:
                try:
                    interpolated_values = enhanced_pattern_based_interpolation(
                        building_data, outlier_indices, interpolation_type='outlier'
                    )
                    
                    for idx, new_value in zip(outlier_indices, interpolated_values):
                        building_data.loc[idx, 'power_consumption'] = new_value
                        
                except Exception as e:
                    pass
        
        # 0값 보간
        zero_mask = building_data['power_consumption'] < 1e-10
        zero_count = zero_mask.sum()
        
        if zero_count > 0:
            zero_indices = building_data[zero_mask].index.values
            
            try:
                interpolated_values = enhanced_pattern_based_interpolation(
                    building_data, zero_indices, interpolation_type='zero'
                )
                
                building_data.loc[zero_indices, 'power_consumption'] = interpolated_values
                
            except Exception as e:
                pass
        
        # 원본 데이터프레임에 반영
        try:
            original_indices = df_interpolated[df_interpolated['building_number'] == building_num].index
            if len(original_indices) == len(building_data):
                df_interpolated.loc[original_indices, 'power_consumption'] = building_data['power_consumption'].values
        except Exception as e:
            pass
    
    print("향상된 스플라인 보간 완료")
    return df_interpolated, {}

def create_weather_features(df):
    """날씨 영향 피처 생성"""
    df['weather'] = 0
    
    rainfall_condition = df['rainfall'] > 0
    rainfall_indices = df[rainfall_condition].index.tolist()
    
    for idx in rainfall_indices:
        for offset in range(-3, 4):
            new_idx = idx + offset
            if 0 <= new_idx < len(df):
                df.loc[new_idx, 'weather'] = 1
    
    print("날씨 영향 피처 생성 완료")
    return df

def create_enhanced_temperature_features(df):
    """향상된 온도 피처"""
    # 3시간 간격 데이터만 사용한 일평균 온도
    three_hour_data = df[df['hour'] % 3 == 0].copy()
    
    enhanced_avg_temp = pd.pivot_table(
        three_hour_data, 
        values='temperature', 
        index=['building_number', 'day', 'month'], 
        aggfunc=np.mean
    ).reset_index()
    enhanced_avg_temp.rename(columns={'temperature': 'enhanced_avg_temp'}, inplace=True)
    
    # 최고/최저/평균 온도
    max_temp = pd.pivot_table(
        df, 
        values='temperature', 
        index=['building_number', 'day', 'month'], 
        aggfunc=np.max
    ).reset_index()
    max_temp.rename(columns={'temperature': 'day_max_temperature'}, inplace=True)
    
    min_temp = pd.pivot_table(
        df, 
        values='temperature', 
        index=['building_number', 'day', 'month'], 
        aggfunc=np.min
    ).reset_index()
    min_temp.rename(columns={'temperature': 'day_min_temperature'}, inplace=True)
    
    mean_temp = pd.pivot_table(
        df, 
        values='temperature', 
        index=['building_number', 'day', 'month'], 
        aggfunc=np.mean
    ).reset_index()
    mean_temp.rename(columns={'temperature': 'day_mean_temperature'}, inplace=True)
    
    # 데이터 병합
    df = df.merge(enhanced_avg_temp, on=['building_number', 'day', 'month'], how='left')
    df = df.merge(max_temp, on=['building_number', 'day', 'month'], how='left')
    df = df.merge(min_temp, on=['building_number', 'day', 'month'], how='left')
    df = df.merge(mean_temp, on=['building_number', 'day', 'month'], how='left')
    
    # 일교차 계산
    df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
    
    print("향상된 온도 통계 피처 생성 완료")
    return df

def create_enhanced_cdh(df):
    """향상된 CDH 계산"""
    def calculate_cdh_sliding(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(np.maximum(0, xs[:(i+1)] - 26)))
            else:
                ys.append(np.sum(np.maximum(0, xs[(i-11):(i+1)] - 26)))
        return np.array(ys)
    
    cdhs = []
    for building_num in range(1, 101):
        building_data = df[df['building_number'] == building_num]
        if len(building_data) > 0:
            building_data = building_data.sort_values('date_time')
            temp_values = building_data['temperature'].values
            cdh_values = calculate_cdh_sliding(temp_values)
            cdhs.extend(cdh_values)
        else:
            cdhs.extend([0] * len(building_data))
    
    df = df.sort_values(['building_number', 'date_time']).reset_index(drop=True)
    df['CDH'] = cdhs
    
    print("향상된 CDH 계산 완료")
    return df

def create_enhanced_power_statistics(train, test):
    """향상된 전력 통계"""
    ratio = np.array([1.0] + [1.0]*2 + [1.0]*2 + [1.0]*2)
    
    train_weighted = train.copy()
    train_weighted['power_consumption_weighted'] = train_weighted.apply(
        lambda row: row['power_consumption'] * ratio[row['day_of_week']], axis=1
    )
    
    # 요일-시간별 통계
    power_mean = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour', 'day_of_week'], 
        aggfunc=np.mean
    ).reset_index()
    power_mean.rename(columns={'power_consumption_weighted': 'day_hour_mean'}, inplace=True)
    
    power_std = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour', 'day_of_week'], 
        aggfunc=np.std
    ).reset_index()
    power_std.rename(columns={'power_consumption_weighted': 'day_hour_std'}, inplace=True)
    
    # 휴일-시간별 통계
    power_holiday_mean = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour', 'holiday'], 
        aggfunc=np.mean
    ).reset_index()
    power_holiday_mean.rename(columns={'power_consumption_weighted': 'holiday_hour_mean'}, inplace=True)
    
    power_holiday_std = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour', 'holiday'], 
        aggfunc=np.std
    ).reset_index()
    power_holiday_std.rename(columns={'power_consumption_weighted': 'holiday_hour_std'}, inplace=True)
    
    # 시간별 통계
    power_hour_mean = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour'], 
        aggfunc=np.mean
    ).reset_index()
    power_hour_mean.rename(columns={'power_consumption_weighted': 'hour_mean'}, inplace=True)
    
    power_hour_std = pd.pivot_table(
        train_weighted, 
        values='power_consumption_weighted', 
        index=['building_number', 'hour'], 
        aggfunc=np.std
    ).reset_index()
    power_hour_std.rename(columns={'power_consumption_weighted': 'hour_std'}, inplace=True)
    
    # 모든 통계를 train과 test에 병합
    datasets = [train, test]
    merge_tables = [
        (power_mean, ['building_number', 'hour', 'day_of_week']),
        (power_std, ['building_number', 'hour', 'day_of_week']),
        (power_holiday_mean, ['building_number', 'hour', 'holiday']),
        (power_holiday_std, ['building_number', 'hour', 'holiday']),
        (power_hour_mean, ['building_number', 'hour']),
        (power_hour_std, ['building_number', 'hour'])
    ]
    
    for i, dataset in enumerate(datasets):
        for table, merge_cols in merge_tables:
            dataset = dataset.merge(table, on=merge_cols, how='left')
        datasets[i] = dataset
    
    print("향상된 전력 통계 피처 생성 완료")
    return datasets[0], datasets[1]

def create_additional_features(df):
    """추가 피처 생성"""
    # 고온다습 조건
    df['hot_humid_condition'] = np.where(
        (df['temperature'] > 26) & (df['humidity'] > 70), 1, 0
    )
    
    # 건물 규모 분류
    def classify_building_size(area):
        if area < 1000:
            return 0  # 소형
        elif area < 10000:
            return 1  # 중형
        else:
            return 2  # 대형
    
    df['building_size_category'] = df['total_area'].apply(classify_building_size)
    
    # 면적당 냉방 비율
    df['cooling_ratio'] = df['cooling_area'] / df['total_area']
    
    print("추가 피처 생성 완료")
    return df

def create_squared_features(df):
    """제곱 피처 생성"""
    # temperature 제곱
    df['temperature_squared'] = df['temperature'] ** 2
    
    # humidity 제곱
    df['humidity_squared'] = df['humidity'] ** 2
    
    print("제곱 피처 생성 완료 (temperature_squared, humidity_squared)")
    return df

def create_comprehensive_additional_features(train, test):
    """종합 추가 피처 생성"""
    print("종합 추가 피처 생성 중...")
    
    for df in [train, test]:
        # 시간순 정렬 (inplace=True로 원본 수정)
        df.sort_values(['building_number', 'date_time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # 1. 기상 데이터 롤링
        df['temp_rolling_3h_mean'] = df.groupby('building_number')['temperature'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df['temp_rolling_6h_max'] = df.groupby('building_number')['temperature'].rolling(6, min_periods=1).max().reset_index(0, drop=True)
        df['humidity_rolling_3h_mean'] = df.groupby('building_number')['humidity'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        
        # 2. 기상 데이터 래그
        df['temp_lag_1h'] = df.groupby('building_number')['temperature'].shift(1)
        df['temp_lag_3h'] = df.groupby('building_number')['temperature'].shift(3)
        df['temp_diff_1h'] = df['temperature'] - df['temp_lag_1h']
        df['humidity_lag_1h'] = df.groupby('building_number')['humidity'].shift(1)
        
        # 3. 상호작용 피처
        df['temp_hour_interaction'] = df['temperature'] * df['hour']
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # 4. 고급 시간 피처
        df['is_morning_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
        df['is_afternoon_peak'] = ((df['hour'] >= 14) & (df['hour'] <= 16)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 20)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 5. 계절 진행 피처
        df['days_from_summer_start'] = (df['date_time'] - pd.to_datetime('2024-06-01')).dt.days
        df['summer_progress'] = df['days_from_summer_start'] / 90.0
        
        # 6. 강수 후 경과 시간
        df['hours_since_rain'] = 0
        for building_num in df['building_number'].unique():
            building_mask = df['building_number'] == building_num
            building_data = df[building_mask].sort_values('date_time')
            
            hours_counter = 999
            hours_list = []
            for _, row in building_data.iterrows():
                if row['rainfall'] > 0:
                    hours_counter = 0
                else:
                    hours_counter += 1
                hours_list.append(min(hours_counter, 168))
            
            df.loc[building_mask, 'hours_since_rain'] = hours_list
    
    print("종합 추가 피처 생성 완료")
    return train, test

def load_and_preprocess_data():
    """데이터 로딩 및 전처리"""
    print("데이터 로딩 중...")
    
    train = pd.read_csv(r'C:\Users\sdh\Desktop\전력\data\train.csv')
    test = pd.read_csv(r'C:\Users\sdh\Desktop\전력\data\test.csv') 
    building_info = pd.read_csv(r'C:\Users\sdh\Desktop\전력\data\building_info.csv')
    
    # 컬럼명 영어로 변경
    train = train.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(°C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    })
    train.drop('num_date_time', axis=1, inplace=True)
    
    test = test.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(°C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity'
    })
    test.drop('num_date_time', axis=1, inplace=True)
    
    building_info = building_info.rename(columns={
        '건물번호': 'building_number',
        '건물유형': 'building_type',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'solar_power_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity'
    })
    
    translation_dict = {
        '건물기타': 'Other Buildings',
        '공공': 'Public',
        '학교': 'University',
        '백화점': 'Department Store',
        '병원': 'Hospital',
        '상용': 'Commercial',
        '아파트': 'Apartment',
        '연구소': 'Research Institute',
        'IDC(전화국)': 'IDC',
        '호텔': 'Hotel'
    }
    
    building_info['building_type'] = building_info['building_type'].replace(translation_dict)
    building_info['solar_power_utility'] = np.where(building_info.solar_power_capacity !='-',1,0)
    building_info['ess_utility'] = np.where(building_info.ess_capacity !='-',1,0)
    
    train = pd.merge(train, building_info, on='building_number', how='left')
    test = pd.merge(test, building_info, on='building_number', how='left')
    
    print("데이터 로딩 완료")
    return train, test, building_info

def create_features(train, test):
    """피처 엔지니어링"""
    print("피처 엔지니어링 시작...")
    
    # datetime 변환 및 기본 시간 피처 생성
    train['date_time'] = pd.to_datetime(train['date_time'], format='%Y%m%d %H')
    test['date_time'] = pd.to_datetime(test['date_time'], format='%Y%m%d %H')
    
    train['hour'] = train['date_time'].dt.hour
    train['day'] = train['date_time'].dt.day
    train['month'] = train['date_time'].dt.month
    train['day_of_week'] = train['date_time'].dt.dayofweek
    
    test['hour'] = test['date_time'].dt.hour
    test['day'] = test['date_time'].dt.day
    test['month'] = test['date_time'].dt.month
    test['day_of_week'] = test['date_time'].dt.dayofweek
    
    # 향상된 스플라인 보간 처리
    train_interpolated, _ = apply_enhanced_spline_interpolation(train)
    
    # 날씨 영향 피처 추가
    train = create_weather_features(train_interpolated)
    test = create_weather_features(test)
    
    # 향상된 온도 통계 피처
    train = create_enhanced_temperature_features(train)
    test = create_enhanced_temperature_features(test)
    
    # 추가 피처 생성
    train = create_additional_features(train)
    test = create_additional_features(test)
    
    # 제곱 피처 생성
    train = create_squared_features(train)
    test = create_squared_features(test)
    
    # 백화점 건물별 특별 휴무 패턴 적용
    train['holiday'] = train.apply(is_building_holiday_updated, axis=1).astype(int)
    test['holiday'] = test.apply(is_building_holiday_updated, axis=1).astype(int)
    
    # 순환 인코딩
    train['sin_hour'] = np.sin(2 * np.pi * train['hour']/23.0)
    train['cos_hour'] = np.cos(2 * np.pi * train['hour']/23.0)
    test['sin_hour'] = np.sin(2 * np.pi * test['hour']/23.0)
    test['cos_hour'] = np.cos(2 * np.pi * test['hour']/23.0)
    
    train['sin_date'] = -np.sin(2 * np.pi * (train['month']+train['day']/31)/12)
    train['cos_date'] = -np.cos(2 * np.pi * (train['month']+train['day']/31)/12)
    test['sin_date'] = -np.sin(2 * np.pi * (test['month']+test['day']/31)/12)
    test['cos_date'] = -np.cos(2 * np.pi * (test['month']+test['day']/31)/12)
    
    train['sin_month'] = -np.sin(2 * np.pi * train['month']/12.0)
    train['cos_month'] = -np.cos(2 * np.pi * train['month']/12.0)
    test['sin_month'] = -np.sin(2 * np.pi * test['month']/12.0)
    test['cos_month'] = -np.cos(2 * np.pi * test['month']/12.0)
    
    train['sin_dayofweek'] = -np.sin(2 * np.pi * (train['day_of_week']+1)/7.0)
    train['cos_dayofweek'] = -np.cos(2 * np.pi * (train['day_of_week']+1)/7.0)
    test['sin_dayofweek'] = -np.sin(2 * np.pi * (test['day_of_week']+1)/7.0)
    test['cos_dayofweek'] = -np.cos(2 * np.pi * (test['day_of_week']+1)/7.0)
    
    # 여름 계절 패턴 추가
    train['summer_cos'] = train['date_time'].apply(summer_cos)
    train['summer_sin'] = train['date_time'].apply(summer_sin)
    test['summer_cos'] = test['date_time'].apply(summer_cos)
    test['summer_sin'] = test['date_time'].apply(summer_sin)
    
    # 월 내 주차 패턴 추가
    train['week_of_month'] = train['date_time'].apply(week_of_month)
    test['week_of_month'] = test['date_time'].apply(week_of_month)
    
    # 향상된 CDH 계산
    train = create_enhanced_cdh(train)
    test = create_enhanced_cdh(test)
    
    # THI, WCT 계산
    train['THI'] = 9/5*train['temperature'] - 0.55*(1-train['humidity']/100)*(9/5*train['humidity']-26)+32
    test['THI'] = 9/5*test['temperature'] - 0.55*(1-test['humidity']/100)*(9/5*test['humidity']-26)+32
    
    train['WCT'] = 13.12 + 0.6215*train['temperature'] - 13.947*(train['windspeed']**0.16) + 0.486*train['temperature']*(train['windspeed']**0.16)
    test['WCT'] = 13.12 + 0.6215*test['temperature'] - 13.947*(test['windspeed']**0.16) + 0.486*test['temperature']*(test['windspeed']**0.16)
    
    # 향상된 전력 사용 통계
    train, test = create_enhanced_power_statistics(train, test)
    
    # 종합 추가 피처 생성
    train, test = create_comprehensive_additional_features(train, test)
    
    train = train.reset_index(drop=True)
    
    print("피처 엔지니어링 완료")
    return train, test

# 데이터 로딩 및 피처 엔지니어링 실행
print("="*60)
print("1단계: 데이터 로딩 및 피처 엔지니어링 시작")
print("="*60)

# 데이터 로딩 및 피처 엔지니어링
train, test, building_info = load_and_preprocess_data()
train, test = create_features(train, test)

# 피처 선택 (기존 피처 + 새로 추가된 피처들)
feature_cols =[
    'temperature', 
    'windspeed', 
    'humidity',
    'total_area', 'cooling_area',
    'sin_hour', 'cos_hour', 
    'sin_date', 'cos_date', 'sin_month', 'cos_month', 
    'sin_dayofweek', 'cos_dayofweek', 
    'holiday',
    'weather',
    'enhanced_avg_temp',
    'day_max_temperature', 
    'day_mean_temperature', 
    'day_min_temperature',
    'day_temperature_range',
    'CDH', 
    'THI', 
    'WCT',
    'day_hour_mean', 'day_hour_std', 
    'holiday_hour_mean', 'holiday_hour_std',
    'hour_mean', 'hour_std',
    'summer_cos', 'summer_sin', 
    'week_of_month',
    'hot_humid_condition',
    'building_size_category',
    'cooling_ratio',
    'rainfall',
    # 새로 추가된 제곱 피처들
    'temperature_squared',
    'humidity_squared'
    # 새로 추가된 피처들
    'temp_rolling_3h_mean', 'temp_rolling_6h_max', 'humidity_rolling_3h_mean',
    'temp_lag_1h', 'temp_lag_3h', 'temp_diff_1h', 'humidity_lag_1h'
]

print(f"사용할 피처: {len(feature_cols)}개")
print("새로 추가된 제곱 피처: temperature_squared, humidity_squared")
print("1단계 완료: train, test, feature_cols 변수 준비됨")
print("="*60)