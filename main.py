#!/usr/bin/python
# -*- coding: utf-8 -*-
import pm4py
from ipywidgets import interact

log_path = "logs/running-example.jsonocel"
ocel = pm4py.read_ocel_json(log_path)


def show_relations_for_eid(eid, type):
    print("Filter by event id")
    df = ocel.relations
    df = df[(df['ocel:eid'] == eid) & (df['ocel:type'] == type)]
    return df


def relations_with_customer(customer):
    print("Filter by event customer name")
    ocel2 = pm4py.filtering.filter_ocel_object_attribute(ocel, "ocel:oid", [customer])
    return ocel2.relations


relations_eid = interact(show_relations_for_eid, eid="1.0", type="customers")
relations_customer = interact(relations_with_customer, customer=ocel.objects[ocel.objects["ocel:type"] == "customers"]["ocel:oid"])
