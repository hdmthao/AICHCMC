import numpy as np
import json
import os
import random
import string
import sys

PATH_COUNTING_RESULTS = "/content/HCMCAIC/counting_info"
PATH_SUBMSSION = "/content/HCMCAIC/submission"
def build_mapping_dictionary():
    file_id = open(PATH_ID_LIST, "r")
    file_list = file_id.read().splitlines()
    my_dict = {}
    for each in file_list:
        each_id = each.split(" ")[0]
        each_filename = each.split(" ")[1][:-4]
        my_dict[each_filename] = each_id
    return my_dict

def write_submission(file_name):
    my_dict = {}
    class_type = {"1@":1, "2@":2, "3@":3, "4@":4, "Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}
    count = 0
    with open(os.path.join(PATH_SUBMSSION, file_name + ".txt"), "w") as result_file:
        for file_counting in os.listdir(PATH_COUNTING_RESULTS):
            print("Num file processing:", count)
            results_counting = np.load(os.path.join(PATH_COUNTING_RESULTS, file_counting), allow_pickle=True)

            vid_name = file_counting[5:-8]
            print("Video name", vid_name)
            results_sort = sorted(results_counting, key = lambda x:int(x[0]))
            print(len(results_sort))
            for each in results_sort:
                result_file.write('{} {} {} {}\n'.format(vid_name, str(each[0]), str(each[4]), str(class_type[each[-1]])))
            count += 1

    print("Save submission to {0}".format(file_name))
if __name__ == "__main__":
    file_name = 'submission'
    write_submission(file_name)
