"""
mapping of exercises to needed information to calculate reps.

<exercise_name> : (

<joint angle threshold @ bottom of the movement>,

<joint angle threshold @ top of the movement>,

<angle behaviour as movement is performed>

)

"""
threshold_dict = {
'curls': (145,50,'converge'),
'incline_press': (30,160,'diverge')
}
