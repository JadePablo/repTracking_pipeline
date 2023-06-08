import pandas as pd
from pymongo import MongoClient

#testing purposes
from transform import test
class invalidExerciseError(Exception):
    pass


def getGroup(exercise: str) -> str:
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
    client = MongoClient('mongodb+srv://allenjade154:pJTxInCYg7oHMjH2@cluster0.pkgjxpn.mongodb.net/')
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

#package the loading process into one function
def upload_data(recordedReps: pd.DataFrame, newRow_group: str) -> None:
    choice = prompt_confirmation(recordedReps)

    if choice:
        print('uploading...')
        load(recordedReps,newRow_group)
    else:
        print('no upload')


def test_loading():
    sample_df = test()
    upload_data(sample_df,'pull')

test_loading()