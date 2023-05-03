#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer
from collections import Counter
import numpy as np
from collections import defaultdict


class ClusteringOCEL:
    def __init__(self, model, ocel, dataset_from_ocel=None, create_dataset_from_ocel=None):
        self.model = model
        self.ocel = ocel
        self.normalizer = Normalizer()
        if create_dataset_from_ocel is None:
            self.create_dataset_from_ocel = self.__create_dataset_from_ocel
        else:
            self.create_dataset_from_ocel = create_dataset_from_ocel
        if dataset_from_ocel is None:
            self.X = self.create_dataset_from_ocel(ocel)
        else:
            self.X = dataset_from_ocel

    def set_model(self, model):
        self.model = model

    def set_create_dataset_from_ocel(self, create_dataset_from_ocel):
        self.create_dataset_from_ocel = create_dataset_from_ocel

    def create_dataset(self):
        self.X = self.create_dataset_from_ocel(self.ocel)

    def set_ocel(self, ocel):
        self.ocel = ocel

    @staticmethod
    def __create_dataset_from_ocel(ocel):
        x = []
        event_ids = list(ocel.events["ocel:eid"])
        object_types = list(set(ocel.objects["ocel:type"]))
        object_type_to_id = {k: i for i, k in enumerate(object_types)}
        activity_to_numeric = {k: i for i, k in enumerate(list(set(ocel.events["ocel:activity"])))}
        for eid in event_ids:
            eid_num = float(eid)
            if eid_num % 1000 == 0:
                print(eid_num)
            counted = Counter(ocel.relations[ocel.relations["ocel:eid"] == eid]["ocel:type"])
            event_vec = np.zeros(len(object_type_to_id) + 1)
            for k in counted.keys():
                vec_id = object_type_to_id[k]
                event_vec[vec_id] += counted[k]
            activity = list(ocel.events[ocel.events["ocel:eid"] == eid]["ocel:activity"])[0]
            event_vec[len(object_type_to_id)] = activity_to_numeric[activity]
            x.append(event_vec)
        return np.array(x)

    def fit(self, normalize=True):
        if normalize:
            x = self.normalizer.transform(self.X)
        else:
            x = self.X
        self.model.fit(x)

    def unique_labels(self):
        return np.unique(self.model.labels_)

    def eid_per_cluster(self):
        clusters = defaultdict(list)
        labels = self.model.labels_
        for i in range(labels.size):
            clusters[labels[i]].append(self.ocel.events.iloc[i]["ocel:eid"])
        return clusters
