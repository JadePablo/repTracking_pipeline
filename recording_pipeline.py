from thresholds import threshold_dict
from extract import extract
from transform import transform
from loading import load

#package the entire pipeline into one process
"""
prompt -> extract process -> transformation process -> loading process
"""
def record():
    done = False

    while not done:
        prompt = input('keep going (y) for yes, (anything) for no: ')

        if prompt == 'y':
            recorded_data = extract()
            transformed_data = transform(recorded_data['data'],recorded_data['name'])
            load(transformed_data)
        else:
            done = True

record()