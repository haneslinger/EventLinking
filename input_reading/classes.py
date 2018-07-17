from collections import defaultdict
class Line:
    def __init__(self, filename =None, event_mention=None, cluster=None):
        self.filename = filename
        self.event_mention =  event_mention
        self.cluster = cluster

    def same_cluster(self, ev2):
        if self.cluster == ev2.cluster:
            return 1
        return 0


class File:
    def __init__(self, filename, lines):
        self.data =  defaultdict()
        self.data[filename]=lines

class Pair:
    def __init__(self, ev1_, ev2_, same_=0, fname_=None):
        self.ev1 = ev1_
        self.ev2 = ev2_
        self.same = int(same_)
        self.fname = fname_
