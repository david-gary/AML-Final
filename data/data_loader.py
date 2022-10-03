# matlab loader for the DREAMER.mat file from the sister data folder

# import scipy.io
# import pandas as pd


# def full_raw_data_load():
#     mat = scipy.io.loadmat('DREAMER.mat')
#     return mat

# # need to add a function that breaks the data down into its respective parts
# # and returns a dictionary of the data


# def load_from_each_key(mat_keys, mat_contents):
#     output_dict = {}
#     for key in mat_keys:
#         output_dict[key] = mat_contents[key]
#     return output_dict

# # contents are kept in the 'DREAMER' key


# def parse_dreamer_data(mat_dreamer):
#     # this function will parse the data into the respective parts
#     # and return a dictionary of the data

#     # convert the data into a pandas dataframe
#     # and return the dataframe
#     return pd.DataFrame(mat_dreamer)


# def main():
#     mat_contents = full_raw_data_load()
#     mat_keys = sorted(mat_contents.keys())
#     mat_dreamer = load_from_each_key(mat_keys, mat_contents)['DREAMER']
#     parse_dreamer_data(mat_dreamer)


# main()

import mat4py
import pandas as pd

mat = mat4py.loadmat('DREAMER.mat')

# convert the data into a pandas dataframe

df = pd.DataFrame(mat['DREAMER'])

print(df)

# convert to csv and save

df.to_csv('DREAMER.csv')
