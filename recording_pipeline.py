from thresholds import threshold_dict
from extract import extract
from transform import transform
from loading import upload_data


def record():
    """
    packages the entire rep-recording pipeline into one process:

    prompt -> extract process -> transformation process -> loading process
    """
    prompt = 'y'
    prompt = input('keep going (y) for yes, (anything) for no: ')

    while prompt == 'y':

        recorded_data = extract()
        transformed_data = transform(recorded_data['data'],recorded_data['name'])
        upload_data(transformed_data,transformed_data.loc[0,'exercise'])
        prompt = input('keep going (y) for yes, (anything) for no: ')

record()