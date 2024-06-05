import pandas as pd

def import_data_drop_na():
    df = pd.read_excel('../data/training_set_rel3.xlsx')
    if df.domain1_score.isna().sum() > 0: 
        df.dropna(subset=['domain1_score'], inplace=True)
    else: 
        pass
    return df


def scale_essay_set1(df):
    score_mapping = {
        'A': (11,12),
        'B': (9,10),
        'C': (8,8),
        'D': (6,7),
        'F': (2,5)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df

def scale_essay_set2(df):
    score_mapping = {
        'A': (5,6),
        'B': (4,4),
        'C': (3,3),
        'D': (2,2),
        'F': (1,1)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df


def scale_essay_set3and4(df):
    score_mapping = {
        'A': (3,3),
        'B': (2,2),
        'C': (1,1),
        'D': (0,0)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df

def scale_essay_set5and6(df):
    score_mapping = {
        'A': (4,4),
        'B': (3,3),
        'C': (2,2),
        'D': (1,1),
        'F': (0,0)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df

def scale_essay_set7(df):
    score_mapping = {
        'A': (21,24),
        'B': (18,20),
        'C': (14,17),
        'D': (9,13),
        'F': (2,8)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df

def scale_essay_set8(df):
    score_mapping = {
        'A': (49,60),
        'B': (40,48),
        'C': (36,39),
        'D': (26,35),
        'F': (10,25)
    }
    def map_score_to_grade(score):
        for grade, (lower_bound, upper_bound) in score_mapping.items():
            if lower_bound <= score <= upper_bound:
                return grade
        return None
    
    df['scaled_grade'] = df['domain1_score'].apply(map_score_to_grade)

    return df


import pandas as pd

def scale_essays(df):
    scaled_df = pd.DataFrame()
    
    for s in df.essay_set.unique():
        if s == 1:
            scaled = scale_essay_set1(df[df.essay_set == s])
        elif s == 2:
            scaled = scale_essay_set2(df[df.essay_set == s])
        elif s in [3, 4]:
            scaled = scale_essay_set3and4(df[df.essay_set == s])
        elif s in [5, 6]:
            scaled = scale_essay_set5and6(df[df.essay_set == s])
        elif s == 7:
            scaled = scale_essay_set7(df[df.essay_set == s])
        elif s == 8:
            scaled = scale_essay_set8(df[df.essay_set == s])
        
        scaled_df = pd.concat([scaled_df, scaled], ignore_index=True)
    
    return scaled_df







