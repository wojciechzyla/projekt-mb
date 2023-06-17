#!/usr/bin/python
# -*- coding: utf-8 -*-
import pm4py
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt

#running-example
log_path = "logs/recruiting.jsonocel"
ocel = pm4py.read_ocel_json(log_path)

def show_basic_information(log_path):
    ocel_ = pm4py.read_ocel_json(log_path)    
    events_number = len(ocel_.relations['ocel:eid'].drop_duplicates().values.tolist())
    objects_number = len(ocel_.relations['ocel:oid'].drop_duplicates().values.tolist())
    object_types = pm4py.ocel_get_object_types(ocel)

    print(f"Your log includes: \n- {events_number} different events")
    print(f"- {objects_number} different objects")
    
    # Object Types
    df = ocel.relations
    numbers = []
    for category in object_types:
        objects_number = len(df[(df['ocel:type'] == category)]['ocel:oid'].drop_duplicates().values.tolist())
        numbers.append(objects_number)
        
    ocel_ = ocel.relations['ocel:type'].drop_duplicates().values.tolist()
    df = pd.DataFrame()
    df["Object category"] = ocel_
    df['Number'] = numbers
    print("\n")
    print(df.reset_index(drop=True).head())

    # Events types
    df_activities = pd.DataFrame()
    ocel_ = pm4py.read_ocel_json(log_path)
    for category in object_types:
        ocel2 = pm4py.ocel_object_type_activities(ocel_)[category]
        df_ocel_ = pd.DataFrame(ocel2)
        df_activities[f"Activities - {category}"] = df_ocel_
        
    print("\n")
    print(df_activities.reset_index(drop=True).head())
    print("\n")


def show_relations_for_eid(eid):
    print("Filter by event id")
    df = ocel.relations
    df = df[(df['ocel:eid'] == eid)]
    #df = df[(df['ocel:eid'] == eid) & (df['ocel:type'] == type)]
    return df

def show_relations_for_eid_and_object_type(eid, type):
    print("Filter by event id and object type")
    ocel2 = pm4py.filtering.filter_ocel_event_attribute(ocel, "ocel:eid", [eid])
    ocel2 = pm4py.filtering.filter_ocel_object_attribute(ocel2, "ocel:type", [type])
    return ocel2.relations

def relations_with_managers(manager):
    print("Filter events by manager name")
    ocel2 = pm4py.filtering.filter_ocel_object_attribute(ocel, "ocel:oid", [manager])
    return ocel2.relations

def customers_orders_timestamp_dependency(customer, timestamp):
    print("Filter by order timestamp and customer")
    ocel2 = pm4py.filtering.filter_ocel_object_attribute(ocel, "ocel:oid", [customer])
    
    return ocel2.relations

def number_of_applicants_according_to_date(date):
    print("Number of applicants depending on the date")
    ocel_ = ocel
    ocel_.get_extended_table().head()
    df = ocel_.get_extended_table()
    df[(df['ocel:activity'] == 'submit application')]
    df["ocel:timestamp"] = pd.to_datetime(df["ocel:timestamp"])
    df["ocel:timestamp"] = df["ocel:timestamp"].dt.date

    df_data_orders = pd.DataFrame()
    df_data_orders["Date"] = df["ocel:timestamp"].drop_duplicates()
    df_data_orders["Applicants"] = df["ocel:timestamp"].map(df.groupby("ocel:timestamp").size())

    print(df_data_orders[df_data_orders["Date"] == date])
    
def number_of_applicants_according_to_date_plot():
    print("Number of orders depending on the date")
    ocel_ = ocel
    ocel_.get_extended_table().head()
    df = ocel_.get_extended_table()
    df[(df['ocel:activity'] == 'submit application')]
    df["ocel:timestamp"] = pd.to_datetime(df["ocel:timestamp"])
    df["ocel:timestamp"] = df["ocel:timestamp"].dt.date

    df_data_orders = pd.DataFrame()
    df_data_orders["Date"] = df["ocel:timestamp"].drop_duplicates()
    df_data_orders["Orders"] = df["ocel:timestamp"].map(df.groupby("ocel:timestamp").size())

    f, ax = plt.subplots(figsize=(12,5))
    plt.bar(df_data_orders["Date"], df_data_orders["Orders"])
    plt.show()

def objects_interaction_oid(oid):
    print("Chosen object interaction summary:")
    ocel_ = pm4py.ocel_objects_interactions_summary(ocel)
    ocel_=ocel_[ocel_["ocel:oid"] == oid]
    print(ocel_)
    
def related_objects_for_event(eid):
    oc = pm4py.ocel_objects_ot_count(ocel)[eid]
    print(f'Related objects for chosen event: {oc}')
    
def show_objects_terminated(obj_type):
    print("Chosen objects type termination:")
    oc = pm4py.filtering.filter_ocel_end_events_per_object_type(ocel, obj_type).get_extended_table()
    print(oc)

#===========TYPE AND ACTIVITIES DEPENDENCY HANDLING==============
def relations_with_object_type_and_activities(type, activity):
    print("Filter by object type and activity")
    ocel2 = pm4py.filtering.filter_ocel_object_attribute(ocel, "ocel:type", [type])
    ocel2 = pm4py.filtering.filter_ocel_event_attribute(ocel2, "ocel:activity", [activity])
    return ocel2.relations

object_type_dropdown = widgets.Dropdown(options=pm4py.ocel_get_object_types(ocel), description='Object type:')
activity_dropdown = widgets.Dropdown(description='Activity:')

def update_activity_options(change):
    # Get the new selected object type
    selected_type = change.new
    # Get the activities for the selected object type
    activities = pm4py.ocel_object_type_activities(ocel)[selected_type]
    # Update the options of the activity dropdown
    activity_dropdown.options = activities
    
object_type_dropdown.observe(update_activity_options, names='value')


#================= FOR IPYNB ================
display_basic_info = interact(show_basic_information, log_path = "logs/recruiting.jsonocel")
relations_eid = interact(show_relations_for_eid_and_object_type, eid="32", type=pm4py.ocel_get_object_types(ocel))
relations_customer = interact(relations_with_managers, manager=ocel.objects[ocel.objects["ocel:type"] == "managers"]["ocel:oid"])
#relations_object_type = interact(relations_with_object_type_and_activities, type=pm4py.ocel_get_object_types(ocel), activity = pm4py.ocel_object_type_activities(ocel)["packages"])
relations_object_type = interact(relations_with_object_type_and_activities, type=object_type_dropdown, activity = activity_dropdown)
number_of_applicants = interact(number_of_applicants_according_to_date, date = ocel.get_extended_table()["ocel:timestamp"].dt.date.drop_duplicates())
plot_number_of_applicants = interact(number_of_applicants_according_to_date_plot)
rel_obj_for_event = interact(related_objects_for_event, eid = '1.0')
when_objects_terminated = interact(show_objects_terminated, obj_type = pm4py.ocel_get_object_types(ocel))





#relations_object_type = interact(relations_with_object_type_and_activities, type=pm4py.ocel_get_object_types(ocel), activity = pm4py.ocel_object_type_activities(ocel)["packages"])

# relations_eid = interact(show_relations_for_eid, eid="1.0", type="customers")
# relations_eid2 = interact(show_relations_for_eid_and_object_type, eid=ocel.relations['ocel:eid'].drop_duplicates().values.tolist()[:20], type=pm4py.ocel_get_object_types(ocel))



                                