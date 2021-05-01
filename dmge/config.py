DATA_PATHA = "../weibohawkes/"
cascades  = "../dataset_weibo.txt"

# DATA_PATHA = "../weibo170w/"
# cascades  = "../weibo170w/dataset_weibo170w.txt"

# DATA_PATHA = "../weibosmp/"
# cascades  = "../weibosmp/dataset_weibo_smp_deephawkes.txt"
#
# DATA_PATHA = "../citation_aps/"
# cascades  = "../dataset_citation.txt"

# generate shortest paths in gen_shortestpath.py
cascade_train  = DATA_PATHA+"/cascade_train.txt"
cascade_val = DATA_PATHA+"/cascade_val.txt"
cascade_test = DATA_PATHA+"/cascade_test.txt"
shortestpath_train = DATA_PATHA+"/shortestpath_train.txt"
shortestpath_val = DATA_PATHA+"/shortestpath_val.txt"
shortestpath_test = DATA_PATHA+"/shortestpath_test.txt"

#gen erate pkl files in preprocess.py
train_pkl = DATA_PATHA+"/data_train.pkl"
val_pkl = DATA_PATHA+"/data_val.pkl"
test_pkl = DATA_PATHA+"/data_test.pkl"
information = DATA_PATHA+"/information.pkl"

# is_weibo True: weibo dataset, is_weibo False: Citation dataset
is_weibo = True
least_num = 10 # lowest num /popularity/retweet in observation time.
up_num = 200

start_hour = 7
end_hour = 19

# for weibo
observation_time = 3600  #  600 10min 3600 1hour
pre_times = [24 * 3600]  # target predicting time (e.g. 24hours)


# for citation aps
# T = 5
# observation_time = 365*T
# pre_times = [20 * 365]

observation_number = 10
time_or_number = 'Time'  # ['Time' or 'Num']

PRETRAIN=False
import math

#parameters

observation = observation_time-1
print("observation time",observation)
n_time_interval = 6
print("the number of time interval:",n_time_interval)
time_interval = math.ceil((observation+1)*1.0/n_time_interval)
print("time interval:",time_interval)
