from dmge import config
import six.moves.cPickle as pickle
import time

def get_hour(time_str, filename):
    hour = None
    try:
        msg_time = int(time_str)
        hour = time.strftime("%H", time.localtime(msg_time))
        hour = int(hour)
    except:
        if '170w' in filename:  # fixed in 11.15, however, in this way, more datasets will be removed
            ts = time.strptime(time_str, '%Y-%m-%d-%H:%M:%S')
            hour = ts.tm_hour
        elif 'castle' in filename:
            # for data castle weibo
            hour = int(time_str[:2])
        elif 'smp' in filename:
            ts = time.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            hour = ts.tm_hour
        else:
            print('wrong time format')
    return hour

def gen_cascade_graph(observation_time, observation_num, pre_times, filename, filename_ctrain, filename_cval, filename_ctest,
                      filename_strain, filename_sval, filename_stest):
    file = open(filename)
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    cascades_total = dict()

    for line in file:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        # print cascadeID
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path):
            print('wrong number of nodes', n_nodes, len(path))

        hour = get_hour(parts[2], filename)

        # print msg_time,hour CasCN 7-20
        # if 6 > hour or hour > 21:
        #     continue

        # to keep the same with
        # if hour <= 7 or hour >= 19:  # 8-18
        #     continue
        if hour <= config.start_hour or hour >= config.end_hour:
            continue

        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        # print(cascadeID)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print(nodes)
                continue
            # print nodes
            time_now = int(p.split(":")[1])

            if config.time_or_number =='Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else: # my addition by observation_num
                if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

            for i in range(len(pre_times)):
                if time_now < pre_times[i]:
                    labels[i] += 1

        # if labels[0]>1000:
        #     continue

        # if len(observation_path) < config.least_num or len(observation_path) > 1000:
        # least_rewteetnum = config.least_num   # 5 or 10
        # if len(observation_path) < least_rewteetnum:
        #     continue

        if len(observation_path) < config.least_num or len(observation_path)>config.up_num:
            continue
        # try:
        #     cascades_total[cascadeID] = msg_time
        # except:
        cascades_total[cascadeID] = hour

    n_total = len(cascades_total)
    print('total:', n_total)
    import operator
    sorted_msg_time = sorted(cascades_total.items(), key=operator.itemgetter(1))
    cascades_type = dict()
    count = 0
    for (k, v) in sorted_msg_time:
        if count < n_total * 1.0 / 20 * 14:
            cascades_type[k] = 1
        elif count < n_total * 1.0 / 20 * 17:
            cascades_type[k] = 2
        else:
            cascades_type[k] = 3
        count += 1

    # for (k, v) in sorted_msg_time:
    #     if count < n_total * 1.0 / 20 * 16:
    #         cascades_type[k] = 1
    #     elif count < n_total * 1.0 / 20 * 18:
    #         cascades_type[k] = 2
    #     else:
    #         cascades_type[k] = 3
    #     count += 1

    print('train data:', len([cid for cid in cascades_type if cascades_type[cid]==1]))
    print('valid data:', len([cid for cid in cascades_type if cascades_type[cid] == 2]))
    print('test data:', len([cid for cid in cascades_type if cascades_type[cid] == 3]))
    num_train, num_valid, num_test = 0, 0, 0


    # to keep the same with CasCN
    # keptids = pickle.load(open(config.DATA_PATHA+'kept_cascade_id.pkl', 'rb'))

    file.close()
    file = open(filename, "r")
    for line in file:
        parts = line.strip('\n').split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]

        if cascadeID not in cascades_type:
            continue

        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path) :  # what hell wrong?
            print('wrong number of nodes', n_nodes, len(path))

        try:
            msg_time = time.localtime(int(parts[2]))
            # print msg_time
            hour = time.strftime("%H", msg_time)
        except:
            hour = int(parts[2][:2])
        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print(nodes)
                continue
            time_now = int(p.split(":")[1])
            if config.time_or_number=='Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else:
                if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

            for i in range(len(pre_times)):
                # print time,pre_times[i]
                if time_now < pre_times[i]:
                    labels[i] += 1

        for i in range(len(labels)):
            labels[i] = str(labels[i] - len(observation_path))

        # for viral / unviral
        # if cascadeID in viral_cid:
        #     labels[0] = str(0)
        # else:
        #     labels[0] = str(1)

        # if not cascadeID in keptids:
        #     continue

        hour = int(hour)
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctrain.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")
            num_train += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")
            num_valid += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3:
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")
            num_test += 1

    print('train', 'test', 'valid', num_train, num_test, num_valid)
    print('total data:', num_valid+num_train+num_test)

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()



def gen_citation_graph(observation_time, observation_num, pre_times, filename, filename_ctrain, filename_cval, filename_ctest,
                      filename_strain, filename_sval, filename_stest):
    file = open(filename)
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    cascades_total = dict()

    for line in file:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        # print cascadeID
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path) and n_nodes+1 !=len(path):
            print('wrong number of nodes', n_nodes, len(path))

        # 以后这一部分单独独立出来成一个模块，可以方便拓展
        # try:
        #     msg_time = int(parts[2])
        #     hour = time.strftime("%H", time.localtime(msg_time))
        #     hour = int(hour)
        # except:
        #     if '170w' in filename:  # fixed in 11.15, however, in this way, more datasets will be removed
        #         ts = time.strptime(parts[2], '%Y-%m-%d-%H:%M:%S')
        #         hour = ts.tm_hour
        #     elif 'castle' in filename:
        #         # for data castle weibo
        #         hour = int(parts[2][:2])

        # if hour <= 7 or hour >= 19: # not such stuff in citation data
        #     continue

        # just for citation
        ts = time.strptime(parts[2], '%Y-%m-%d')
        hour = ts.tm_mon

        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print(nodes)
                continue
            # print nodes
            time_now = int(p.split(":")[1])

            if config.time_or_number =='Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else: # my addition by observation_num
                if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

            for i in range(len(pre_times)):
                if time_now < pre_times[i]:
                    labels[i] += 1

        # if labels[0]>1000:
        #     continue
        #
        # if len(observation_path) < 10 or len(observation_path) > 1000:
        # least_rewteetnum = config.least_num   # 5 or 10
        # least_rewteetnum = config.least_num
        # if len(observation_path) < least_rewteetnum:
        #     continue
        if len(observation_path) < config.least_num or len(observation_path)>config.up_num:
            continue

        # try:
        #     cascades_total[cascadeID] = msg_time
        # except:
        cascades_total[cascadeID] = hour

    n_total = len(cascades_total)
    print('total:', n_total)
    import operator
    sorted_msg_time = sorted(cascades_total.items(), key=operator.itemgetter(1))
    cascades_type = dict()
    count = 0
    # split 15% TEST 15% VALID 70% TRAIN
    for (k, v) in sorted_msg_time:
        if count < n_total * 1.0 / 20 * 14:
            cascades_type[k] = 1
        elif count < n_total * 1.0 / 20 * 17:
            cascades_type[k] = 2
        else:
            cascades_type[k] = 3
        count += 1

    # split 10% TEST 10% VALID 80% TRAIN
    # for (k, v) in sorted_msg_time:
    #     if count < n_total * 1.0 / 20 * 16:
    #         cascades_type[k] = 1
    #     elif count < n_total * 1.0 / 20 * 18:
    #         cascades_type[k] = 2
    #     else:
    #         cascades_type[k] = 3
    #     count += 1

    # 42186
    #
    print('train data:', len([cid for cid in cascades_type if cascades_type[cid]==1]))
    print('valid data:', len([cid for cid in cascades_type if cascades_type[cid] == 2]))
    print('test data:', len([cid for cid in cascades_type if cascades_type[cid] == 3]))


    file.close()
    file = open(filename, "r")
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        if n_nodes != len(path) and n_nodes+1 != len(path):
            print('wrong number of nodes', n_nodes, len(path))


        ts = time.strptime(parts[2], '%Y-%m-%d')
        hour = ts.tm_mon

        observation_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            nodes_ok = True
            for n in nodes:
                if int(n) == -1:
                    nodes_ok = False
            if not (nodes_ok):
                print(nodes)
                continue
            time_now = int(p.split(":")[1])
            if config.time_or_number=='Time':
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            else:
                if len(observation_path) <= observation_num+1 and time_now < pre_times[0]:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")

            for i in range(len(pre_times)):
                # print time,pre_times[i]
                if time_now < pre_times[i]:
                    labels[i] += 1

        for i in range(len(labels)):
            labels[i] = str(labels[i] - len(observation_path))

        # for viral / unviral
        # if cascadeID in viral_cid:
        #     labels[0] = str(0)
        # else:
        #     labels[0] = str(1)

        hour = int(hour)
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctrain.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3:
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(
                cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                    edges) + "\t" + " ".join(labels) + "\n")

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()


if __name__ == "__main__":
    print('yes')
    start = time.time()
    observation_time = config.observation_time
    observation_num = config.observation_number
    # pre_times = [24 * 3600]
    pre_times = config.pre_times

    weibo = config.is_weibo
    if weibo:
        gen_cascade_graph(observation_time, observation_num, pre_times, config.cascades, config.cascade_train, config.cascade_val,
                          config.cascade_test,
                          config.shortestpath_train, config.shortestpath_val, config.shortestpath_test)

    else:
        gen_citation_graph(observation_time, observation_num, pre_times, config.cascades, config.cascade_train, config.cascade_val,
                          config.cascade_test,
                          config.shortestpath_train, config.shortestpath_val, config.shortestpath_test)

    print('total time in preprocessing:', time.time()-start)