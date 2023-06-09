#not packaged wtih python (install dependencies before use)
from pymongo import MongoClient
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

class invalidExerciseError(Exception):
    pass


def getGroup(exercise: str) -> str:
    """
    Find which group (push , pull , legs) that 'exercise' belongs to

    Args:
        exercise (str): name of exercise.

    Returns:
        str: the group that the exercise belongs in.
    """

    pull_exercises = ['pullup', 'row', 'rear-delt-fly', 'curls', 'hammer_curl']
    push_exercises = ['dip', 'incline_press', 'shoulder_press', 'lateral_raise']
    leg_exercises = ['standing_calf_raise', 'rdl', 'bulgarian_split_squat']

    if exercise in pull_exercises:
        return 'pull'

    if exercise in push_exercises:
        return 'push'

    if exercise in leg_exercises:
        return 'legs'

    else:
        raise invalidExerciseError(
            f"The exercise '{exercise}' is not a a recognized exercise. This is all I know: {pull_exercises} , {push_exercises} , {leg_exercises}")


def load(recordedReps: pd.DataFrame, newRow_group: str) -> None:
    """
    upload the 'recordedReps' dataframe into database.

    Args:
        recordedReps (pd.DataFrame): meta and timing information about the recorded reps.
    """
    connection_string = os.getenv('CONNECTION_STRING')
    client = MongoClient(connection_string)
    db = client['Cluster0']
    collection = db[newRow_group]
    mongodb_readable = recordedReps.to_dict(orient='records')
    try:
        # Insert the data into the collection
        result = collection.insert_many(mongodb_readable)

        # Check if the insert was successful
        if result.inserted_ids:
            print("Insert was successful.")
        else:
            print("Insert failed.")
    except Exception as e:
        print("An error occurred:", e)

def prompt_confirmation(recordedReps: pd.DataFrame) -> bool:

    print (recordedReps)
    response = input('Here are the recorded reps, if its accurate, upload it (u), if you want terminate it, delete it (d))')

    while response != 'u' and response != 'd':
        response = input(
            'Here are the recorded reps, if its accurate, upload it (u), if you want terminate it, delete it (d)): ')

    if response == 'u':
        return True

    return False


def upload_data(recordedReps: pd.DataFrame, exercise: str) -> None:

    perceivedGroup = getGroup(exercise)
    choice = prompt_confirmation(recordedReps)

    if choice:
        print('uploading...')
        load(recordedReps,perceivedGroup)
    else:
        print('no upload')

