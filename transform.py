#import dependencies of these modules before use
import pandas as pd
import numpy as np
import random

#these come packaged with python
from datetime import date
from typing import List, Tuple
from thresholds import threshold_dict

def add_position_labels(input_df: pd.DataFrame,exercise_name: object) -> pd.DataFrame:
    """
    add the movement stages (bottom,top,ascent,descent) to the recordings.

    Args:
        input_df (pd.DataFrame): dataframe of raw recordings
        exercise_name (pd.DataFrame): associated performed exercise

    Returns:
        pd.DataFrame: annotated raw recordings

    """

    #look up the exercises bottom and top thresholds
    bottom = threshold_dict[exercise_name][0]
    top = threshold_dict[exercise_name][1]

    labelled_input = input_df

    bottom_reached = False
    top_reached = False


    #lambda function using the thresholds to distinguish 'top' and 'bottom' stages
    position_function = lambda angle: 'bottom' if angle >= bottom else 'top' if angle <= top else None

    # Apply the lambda function to create the 'position' column
    labelled_input['position'] = labelled_input['angle'].apply(position_function)

    last_visited = None

    #annotate recordings that captures an 'ascent' (rows between bottom and top)
    #do the same for recordings that captures a 'descent' (rows between top and bottom)
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
def get_stages_converging(labelled_df: pd.DataFrame) -> dict:
    """
    extrapolate the reps and timing information from the annotated raw recording

    Args:
        labelled_df (pd.DataFrame): annotated raw recording

    Returns:
        dict:
            - <key> (string) : the stage of the rep (bottom,top,descent,ascent)
            - <value> (List<tuple>): tuples of start and end times of each detected stage
    """


    #get the start and end times of each bottom 'island' in the df
    consecutive_bottoms = []
    start_time = None
    prev_position = None

    for index, row in labelled_df.iterrows():
        if row['position'] == 'bottom':
            if prev_position != 'bottom':
                start_time = row['time']
        elif row['position'] != 'bottom' and prev_position == 'bottom':
            end_time = labelled_df.at[index - 1, 'time']
            consecutive_bottoms.append((start_time, end_time))

        prev_position = row['position']

    #get the start and end times of each top 'island' in the df
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

    #get the start and end times of each ascent 'island' in the df
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

    #get the start and end times of each descent 'island' in the df
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

    #all reps end with a descent, check if there are consecutive descents at the end
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
    infer time attributes of the rep using the durations of (bottom,top,ascent,descent) from 'stages'

    Args:
        stages (dict): pairs of stage_names and list of tuples that map the indices to the [start,end] of that stage
        exercise_name (string): name of exercise associated with reps

    Returns:
        pd.DataFrame: dataframe where each row is a rep, and each column are the time attributes (as shown directly below) of each rep
    """

    reps = pd.DataFrame(columns=['start','end','total','ascent','descent','top_pause','bottom_pause'])

    #calculate the start of the rep and time spent at the bottom
    for duration in stages['bottoms']:
        new_entry_info = {
            'start': duration[0],
            'bottom_pause':duration[1] - duration[0]
        }
        new_entry = pd.DataFrame(new_entry_info,index=[len(reps)])

        reps = pd.concat([reps,new_entry], ignore_index=True)

    #calculate time spent during ascent
    for i in range(len(stages['ascents'])):
        reps.at[i,'ascent'] = stages['ascents'][i][1] - stages['ascents'][i][0]

    #calculate time spent during descent
    for i in range(len(stages['descents'])):
        reps.at[i,'descent'] = stages['descents'][i][1] - stages['descents'][i][0]
        #total = descents[1] - start in df
        reps.at[i,'total'] = stages['descents'][i][1] - reps.loc[i,'start']
        reps.at[i, 'end'] = stages['descents'][i][1]

    #calculate time spent at the top
    for i in range(len(stages['tops'])):
        reps.at[i,'top_pause'] = stages['tops'][i][1] - stages['tops'][i][0]

    #add the exercise name and date of when it was performed
    reps['exercise'] = exercise_name
    reps['date'] = date.today().strftime("%Y-%m-%d")
    #feed me into the loading script as is
    return(reps)

def transform(raw_data: pd.DataFrame , exercise_name: pd.DataFrame) -> pd.DataFrame:
    """
    bundles the entire transformation sequence into one process

    Args:
        raw_data (pd.DataFrame): raw_recordings from the extract process
        exercise_name (string): name of exercise associated with reps

    Returns:
        pd.DataFrame: information about the reps
    """
    labelled_data = add_position_labels(raw_data,exercise_name)
    stages = get_stages_converging(labelled_data)
    result = get_reps(stages,exercise_name)

    return result