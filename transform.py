import pandas as pd
import numpy as np
import random

from datetime import date
from typing import List, Tuple
from thresholds import threshold_dict
"""
one rep follows the stages: bottom,top,bottom

how to extrapolate info per rep:
start_time = 
"""
def add_position_labels(input_df: pd.DataFrame,exercise_name: object) -> int:
    bottom = threshold_dict[exercise_name][0]
    top = threshold_dict[exercise_name][1]

    labelled_input = input_df

    bottom_reached = False
    top_reached = False

    # Define a lambda function to determine the position based on the 'angle' value
    position_function = lambda angle: 'bottom' if angle >= bottom else 'top' if angle <= top else None

    # Apply the lambda function to create the 'position' column
    labelled_input['position'] = labelled_input['angle'].apply(position_function)

    last_visited = None

    for index, row in labelled_input.iterrows():
        if row['position'] == "bottom":
            last_visited = "bottom"
        elif row['position'] == "top":
            last_visited = "top"
        else:
            if last_visited == "bottom":
                labelled_input.at[index, 'position'] = "ascent"
            else:
                labelled_input.at[index, 'position'] = "descent"


    return labelled_input
def get_stages(labelled_df: pd.DataFrame) -> List[Tuple[int, int]]:
    consecutive_bottoms = []
    start_time = None
    prev_position = None

    # Get the time values of bottoms
    for index, row in labelled_df.iterrows():
        if row['position'] == 'bottom':
            if prev_position != 'bottom':
                start_time = row['time']
        elif row['position'] != 'bottom' and prev_position == 'bottom':
            end_time = labelled_df.at[index - 1, 'time']
            consecutive_bottoms.append((start_time, end_time))

        prev_position = row['position']

    # Get the time values of tops
    consecutive_tops = []
    start_time = None
    prev_position = None

    for index, row in labelled_df.iterrows():
        if row['position'] == 'top':
            if prev_position != 'top':
                start_time = row['time']
        elif row['position'] != 'top' and prev_position == 'top':
            end_time = labelled_df.at[index - 1, 'time']
            consecutive_tops.append((start_time, end_time))

        prev_position = row['position']

    # Get the time values of ascents
    consecutive_ascents = []
    start_time = None
    prev_position = None

    for index, row in labelled_df.iterrows():
        if row['position'] == 'ascent':
            if prev_position != 'ascent':
                start_time = row['time']
        elif row['position'] != 'ascent' and prev_position == 'ascent':
            end_time = labelled_df.at[index - 1, 'time']
            consecutive_ascents.append((start_time, end_time))

        prev_position = row['position']

    # Get the time values of descents
    consecutive_descents = []
    start_time = None
    prev_position = None

    for index, row in labelled_df.iterrows():
        if row['position'] == 'descent':
            if prev_position != 'descent':
                start_time = row['time']
        elif row['position'] != 'descent' and prev_position == 'descent':
            end_time = labelled_df.at[index - 1, 'time']
            consecutive_descents.append((start_time, end_time))

        prev_position = row['position']

    # Check if there are consecutive descents at the end
    if prev_position == 'descent':
        end_time = labelled_df.at[index, 'time']
        consecutive_descents.append((start_time, end_time))

    return {
        'bottoms': consecutive_bottoms,
        'ascents': consecutive_ascents,
        'tops': consecutive_tops,
        'descents': consecutive_descents
    }

def get_reps(stages:dict,exercise_name: str) -> pd.DataFrame:
    """

    :param stages: list of tuples that map the indices to the [start,end] of that stage
    :return: dataframe containing rep data extrapolated from the input dictionary: stages.

    """
    reps = pd.DataFrame(columns=['start','end','total','ascent','descent','top_pause','bottom_pause'])
    for duration in stages['bottoms']:
        new_entry_info = {
            'start': duration[0],
            'bottom_pause':duration[1] - duration[0]
        }
        new_entry = pd.DataFrame(new_entry_info,index=[len(reps)])

        reps = pd.concat([reps,new_entry], ignore_index=True)


    for i in range(len(stages['ascents'])):
        reps.at[i,'ascent'] = stages['ascents'][i][1] - stages['ascents'][i][0]

    for i in range(len(stages['descents'])):
        reps.at[i,'descent'] = stages['descents'][i][1] - stages['descents'][i][0]
        #total = descents[1] - start in df
        reps.at[i,'total'] = stages['descents'][i][1] - reps.loc[i,'start']
        reps.at[i, 'end'] = stages['descents'][i][1]

    for i in range(len(stages['tops'])):
        reps.at[i,'top_pause'] = stages['tops'][i][1] - stages['tops'][i][0]

    reps['exercise'] = exercise_name
    reps['date'] = date.today()
    #feed me into the loading script as is
    return(reps)

def transform(raw_data: pd.DataFrame , exercise_name: pd.DataFrame):
    labelled_data = add_position_labels(raw_data,exercise_name)
    stages = get_stages(labelled_data)
    result = get_reps(stages,exercise_name)

    return result

def test() -> pd.DataFrame:
    # sequence = [(160, 1), (70, 1), (15, 1), (60, 1)]
    # repetitions = 7
    #
    #
    # # Initialize an empty list to store the rows
    # rows = []
    #
    # # Initialize the initial time value
    # time = 1
    #
    # # Generate the rows based on the sequence and repetitions
    # for _ in range(repetitions):
    #     for angle, _ in sequence:
    #         for _ in range(5):
    #             rows.append({'angle': angle, 'time': time})
    #             time += 1
    #
    # # Create the DataFrame
    # df = pd.DataFrame(rows)

    test_df = pd.read_csv('curls_test.csv')
    pd.set_option('display.max_rows',None)
    pd.set_option('display.max_columns',None)
    print( transform(test_df,'curls'))
    # labelled_data = add_position_labels(test_df,'curls')
    #
    # result = get_stages(labelled_data)
    # for key,value in result.items():
    #     print(key,value)
    #
    # pd.set_option('display.max_rows',None)
    #
    # print(get_reps(result))


test()

