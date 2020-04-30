import json
from pprint import pprint
import pandas as pd
import re


def clean_text(raw_text: str):
    if raw_text is None:
        return ''

    raw_text = raw_text.replace('\n', ' ')
    return re.sub(re.compile('<.*?>'), '', raw_text)


def read_json_as_df(path: str) -> pd.DataFrame:
    json_data = []

    with open(path, 'r') as file:

        for line in file:
            data = json.loads(line)
            json_data.append([clean_text(data['post'].get('body', None)),
                              data['priority']])

    df = pd.DataFrame(data=json_data, columns=('text', 'priority'))

    return df


