from typing import Any, Callable, Literal, Tuple
import numpy as np
import pandas as pd
from pm4py.objects.ocel.obj import OCEL
from Levenshtein import distance as lev
# own files
import clustering as clustering
import distance_calculation as distance_calculation
import functions as functions
from clusteval_fix import clusteval

# publish constants from clustering!

# default name of columns
OCEL_COL_NAME_OID           = 'ocel:oid'
OCEL_COL_NAME_EID           = 'ocel:eid'
OCEL_COL_NAME_OBJECT_TYPE   = 'ocel:type'
OCEL_COL_NAME_TIMESTAMP     = 'ocel:timestamp'
OCEL_COL_NAME_ACTIVITY      = 'ocel:activity'
OCEL_COL_NAME_CONTROL_FLOW  = 'cflow'
OCEL_COL_NAME_CLUSTER       = 'cluster'

# load from clustering module
OCEL_CLUSTER_ALGORITHMS = clustering.CLUSTER_ALGORITHMS # are the same as possible in the cluster module

# default values for data-types
DEFAULT_VALUES = {
    'object': '', # assuming string
    'float64': np.float64(0.0)
}

# default comparisons for data-types for distance measuring
DEFAULT_COMPARISONS = {
    'object': lambda x,y: x!=y,
    'float64': lambda x,y: abs(x-y)
}

# considers the data. If a column is completly NaN it returns for that column NoneType. Fo strings the compare function is just 1 for inequal and 0 for equal. for numbers its the distance
def ocel_get_attr_def(ocel: OCEL) -> list[dict['name': str, 'weight': float, 'default': any, 'compare_func': Callable[[Any, Any], float]]]:
    """
    Returns information for every attribute of the objects Dataframe (```ocel.objects```) of the given OCEL-instance.

    Currently the function only differs between objects and float64 types of the attributes.

    Parameters
    ----------
    ocel : OCEL
        The OCEL-element to get the attribute definitions.

    Returns
    -------
    list
        A list of dictionaries (one dict per attribute of ocel.objects leaving out ocel:oid and ocel:type) which consist of the following information:
        name: Name of the Dataframe-column
        weight: Always 1
        default: depending on the Datatype of the column. Can be 0, NaN, '' (empty string), etc.
        compare_func: Callable returning float which can be used to compare two values of this attribute (euclidean distance or levenshtein distance)
    """
    res = []
    # getting all types
    types = ocel.objects.dtypes
    # filtering to types which we can handle
    types = types[types.map(lambda x: x in list(DEFAULT_VALUES.keys()))]
    # removing id column, because not used for clustering or anything
    types = types.drop(labels=[OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE], errors='ignore').to_dict()
    for label in types.keys():
        col_type = str(types[label])
        res.append({
            'name': label,
            'weight': 1,
            'default': DEFAULT_VALUES[col_type],
            'compare_func': DEFAULT_COMPARISONS[col_type]
        })
    return res

def ocel_object_cluster_to_event_cluster(
    ocel: OCEL, 
    object_clusters: pd.DataFrame, 
    mode: Literal['all', 'existence'],
    OCEL_COL_NAME_CLUSTER: str = OCEL_COL_NAME_CLUSTER
) -> pd.DataFrame:
    """
    Returns a filtered ```ocel.events``` Dataframe. The parameter ```object_clusters``` - 
    which is expected to be a subset of ocel.objects with an additional cluster column - 
    defines multiple object-clusters which are used to create a corresponding event-clusters.

    The cluster of an event is determined by the related objects that are in the table object_clusters. Via ```ocel.relations``` all given objects in the DataFrame 
    ```object_clusters``` are joined to the related events from ```ocel.events```. Then the clusters of related objcets are assigned to the event data.
    
    It is possible that this function returns less distinct cluster-values than given in the ```object_clusters``` DataFrame. If a object-instance is not related to
    any event from ```ocel``` no event is assigned to the cluster of this object-instance.

    Parameters
    ----------

    ocel: pm4py.objects.ocel.obj
        Object centric event log on which the function will be applied on.

    object_clusters: pandas.DataFrame
        subset of ```ocel.objects``` which defines the clusters of objects. This DataFrame needs to have the columns 'ocel:oid', 'ocel:type'
        and a additional column with the name definde in the parameter ```OCEL_COL_NAME_CLUSTER```.

    mode: Literal['all', 'existence']
        Defines the exact mode of assigning events to clusters. ```mode == 'all'```: An event is only assigned to a cluster if its cluster values are unique. This means that every related object from object_clusters has to be in the same cluster. 
        If objects related to an event are in different clusters, the event is not added to the returning DataFrame at all. ```mode == 'existence'```: An event may be assigned to multiple clusters. It is added to every cluster any related object is assigned to.

    OCEL_COL_NAME_CLUSTER: str
        Defines the name of the column which contains the cluster-name in the DataFrame ```object.clusters```.

    Returns
    -------
    pandas.DataFrame:
        DataFrame containing all events (```ocel.events```) enriched by a cluster column. Depending on the selected ```mode``` events may not be unique (existence mode). 
    """

    # Defining mandatory columns
    COLS = [OCEL_COL_NAME_CLUSTER, OCEL_COL_NAME_OBJECT_TYPE, OCEL_COL_NAME_OID]
    if not all(col in object_clusters.columns for col in COLS):
        raise Exception('One mandatory column is missing. Ensure that object_clusters contains every of the following columns: "' + '", "'.join(COLS) + '"')

    # Joining clusters to event data via ocel:type and ocel:oid of objects
    JOIN_COLS = [OCEL_COL_NAME_OBJECT_TYPE, OCEL_COL_NAME_OID]
    clusters_indexed = object_clusters.set_index(JOIN_COLS)

    # Joining the relations (event-ids) to the cluster values based on given relations between objects and events.
    event_cluster_data = ocel.relations.join(clusters_indexed, on=JOIN_COLS, how='inner')
    # duplicates can be dropped, as the information that a event is related to a cluster x or y is relevant not how "often" this is the case.
    event_cluster_data = event_cluster_data[[OCEL_COL_NAME_EID, OCEL_COL_NAME_CLUSTER]].drop_duplicates()
    
    if mode == 'all':
        event_cluster_data = event_cluster_data.groupby(OCEL_COL_NAME_EID).agg({OCEL_COL_NAME_CLUSTER: functions.series_single_unique_val_or_nan})
        event_cluster_data.dropna(inplace=True)
    else:
        event_cluster_data.set_index(OCEL_COL_NAME_EID, inplace=True) # groupby in all branch already does the indexing

    # Adding event data, inner join because only events that exists should be considered and only events that are assigned to a cluster.
    # Depending on the object type used for clustering this can be a much smaller set of events than in the original graph.
    res = ocel.events.join(event_cluster_data, on=OCEL_COL_NAME_EID, how='inner').reset_index(drop=True)

    return res


# you may add information to ocel.objects table for using it in the attr_definition. control-flow is used automatically
def ocel_cluster_by_objects(
    ocel: OCEL, 
    object_type: str, 
    event_assignment_mode: Literal['all', 'existence'],
    attr_def: list[dict['name': str, 'weight': float, 'default': any, 'compare_func': Callable[[Any, Any], float]]],
    clustering_algorithm: OCEL_CLUSTER_ALGORITHMS = 'kmeans',
    evaluate: Literal['silhouette', 'dbindex', 'derivative'] = 'silhouette',
    max_cluster_count: int = -1, # -1 means auto detect (up to itemcount - 1). potential very high runtime complexity
    cluster_count: int = -1, # -1 means auto detect in general (up to max_cluster_count)
    control_flow_active: bool = True, # if true the control flow for every object is determined and used for the distance calculation
    control_flow_weight: float = 1.0, # defines the weight of the control flor
    control_flow_compare_func: Callable[[Any, Any], float] = lev, # default is levenshtein distance. 
    control_flow_as_letters: bool = True # defines that the control-flow is translated to unique letters. this is faster to process internally. if a custom compar_function is used, this parameter may be set to false. then two lists are given to the compare function
) -> Tuple[list[OCEL], np.matrix, clusteval]:
    """
    Clusters an input OCEL to multiple OCEL by ```ocel.objects``` via measuring distances between every istance pair of ```ocel.objects```.

    Therefor objects are filtered by the column ```ocel:type``` in ```ocel.objects```. The function calculates every distance pairwise and then calculate the clustering.

    Parameters
    ----------
    ocel: pm4py.objects.ocel.obj
        Object centric event log to use by the algorithm.

    object_type: str,
        Object type (```ocel:type``` in ```ocel.objects```) to apply the clustering on.

    event_assignment_mode: Literal['all', 'existence']
        Used for assigning events to the retrieved object clusters. For further information read the docs for the function ```ocel_object_cluster_to_event_cluster``` which is used internally.

    attr_def: list[dict['name': str, 'weight': float, 'default': any, 'compare_func': Callable[[Any, Any], float]]]
        Definition of ```ocel.objects``` attributes including a column name, a weight for the overall distance measure, a default value for NaN values 
        and a comparison function for measunrng the distance between two object instances.

    clustering_algorithm: OCEL_CLUSTER_ALGORITHMS = 'kmeans',
        Selection of algorithm to use for clustering of the objects by theire distances.

    evaluate: Literal['silhouette', 'dbindex', 'derivative'] = 'silhouette'
        Defines the evaluation mode of the clustering for finding the best k for clustering.

    max_cluster_count: int = -1
        Defines the maximum cluster-count. Up to this number the algorithm tries to find the best k for clustering. If this number is set to -1, the algorithm optimizes whilst
        using a k up to object count minus 1. If cluster_count is set to a value greater or equal to two, that fixes cluster_count is used.

    cluster_count: int = -1
        Defines the cluster-count. This number is used as fixed cluster-count. If this number is equal to -1, the best k is searched according the parameters ```max_cluster_count``` and ```evaluate```.

    control_flow_active: bool = True
        Defines if the control-flow of event actions is calculated for each object instance and if the distance between the control-flow of every object-instance pair is used
        for the overall distance measuring.

    control_flow_weight: float = 1.0
        Weight of the calculated control-flow distance. This is used like 'weight' for every attribute definition in the Parameter ```attr_def```.

    control_flow_compare_func: Callable[[Any, Any], float] = Levenshtein.distance
        Defines the distance function used for the control-flow. Needs two be a two-ary function with a float as result.

    control_flow_as_letters: bool = True
        Defines if the control flow (which is a list of strings of action names) is transformed to a string where every occuring letter represents uniqueliy an action name.
    
    Returns
    -------

    list[OCEL]:
        A list of OCEL objects which each are a representation of one cluster.

    numpy.matrix:
        A matrix which represents the resulting distance matrix for the input objects.

    clusteval:
        A clustevcal object which can be used to generate information about how bad or good the different k's used for clustering fit the data. 
    
    """

    # VALIDATING inputs
    if object_type not in ocel.objects[OCEL_COL_NAME_OBJECT_TYPE].unique():
        raise Exception('Error in ocel_cluster_by_objects: object_type "' + object_type + '" not present in the data.')
    # setting clustering_algo to enum element
    # cluster_algo = clustering.cluster_algo_name_to_algo_enum[clustering_algorithm]


    # DISTANCE CALCULATION
    # getting the relevant information for distance calculation (basically the ocel.objects table)
    if control_flow_active:
        # adding attribute definition for control_flow
        attr_def.append({'name' : OCEL_COL_NAME_CONTROL_FLOW, 'weight': control_flow_weight, 'default': '' if control_flow_as_letters else [], 'compare_func': control_flow_compare_func})
        object_data = ocel_get_control_flow_per_object(ocel=ocel, object_types=[object_type], control_flow_col_name=OCEL_COL_NAME_CONTROL_FLOW)
        if control_flow_as_letters:
            letter_map = functions.get_to_char_map(ocel.relations[ocel.relations[OCEL_COL_NAME_OBJECT_TYPE] == object_type][OCEL_COL_NAME_ACTIVITY].unique())
            object_data[OCEL_COL_NAME_CONTROL_FLOW] = object_data[OCEL_COL_NAME_CONTROL_FLOW].apply(lambda cf_list: ''.join(list(map(lambda activity: letter_map[activity], cf_list))))
    else:
        object_data = ocel.objects[ocel.objects[OCEL_COL_NAME_OBJECT_TYPE] == object_type]
    # dropping typing column, setting index to oid, dropping full NaN-columns
    object_data = object_data.drop(OCEL_COL_NAME_OBJECT_TYPE, axis=1).set_index(OCEL_COL_NAME_OID).dropna(how='all', axis=1)
    # removing non relevant columns before calculating the distance
    relevant_object_data = object_data[[attr['name'] for attr in attr_def if attr['weight'] != 0 and attr['name'] in object_data.columns]]
    # creating a attribute dictionary from attribute definition list
    attr_def_dict = {attr['name']: attr for attr in attr_def}
    # filling nan values in relevant columns
    for col in list(relevant_object_data.columns):
        nan_val = attr_def_dict[col]['default']
        relevant_object_data[col] = relevant_object_data[col].fillna(nan_val)
    # calculating distance matrix
    distance_matrix = distance_calculation.df_distance_matrix(relevant_object_data, attr_def=attr_def)
    # create objects table from data with index (for mapping to groups later)
    objects = relevant_object_data.reset_index()[[OCEL_COL_NAME_OID]]
    
    # CLUSTERING
    # ensure useful max_cluster_count
    if cluster_count < 2:
        k = max_cluster_count if max_cluster_count < distance_matrix.shape[0] and max_cluster_count >= 2 else distance_matrix.shape[0] - 1
    else:
        k = cluster_count
    # apply auto-sized-clustering
    # cluster_evaluation contains the evaluation of the clusters.
    clusters, cluster_evaluation = clustering.find_clusters(distance_matrix, clustering_algorithm, evaluate, k, auto_k=(cluster_count < 2))

    # assigning clusters to objects
    objects[OCEL_COL_NAME_CLUSTER] = clusters # because they are ordered the same way
    objects[OCEL_COL_NAME_OBJECT_TYPE] = object_type # next functions expects this.

    # EVENT ASSIGNMENT:
    events_clustered = ocel_object_cluster_to_event_cluster(
        ocel=ocel, 
        object_clusters=objects, 
        mode=event_assignment_mode, 
        OCEL_COL_NAME_CLUSTER=OCEL_COL_NAME_CLUSTER
    )
    # !!! -> some clusters are dropped (the ones where no object from the cluster is related to any event)

    # ocel creating from event_clusters
    sub_ocels = ocel_group_by_events(ocel, events=events_clustered, objects=objects, group_col_name=OCEL_COL_NAME_CLUSTER)
    
    return list(sub_ocels.values()), distance_matrix, cluster_evaluation

# returns full ocel.object table. adds a column namend as defined in controlg_flow_col_name with the ordered occurences of activity names as list.
def ocel_get_control_flow_per_object(
    ocel: OCEL, 
    object_types: list[str] = None, # use for potential speedup. 
    control_flow_col_name: str = OCEL_COL_NAME_CONTROL_FLOW
) -> pd.DataFrame:
    """
    Retrieves control-flow for every object that is of one of the defined types in ```object_types```. 

    Parameters
    ----------
    ocel: OCEL
        The object-centric event log to use by the function.

    object_types: list[str] = None
        Defines the list of object_types that should be considered for generating the control-flow. If you would like to calculate the control_flow for every object_type
        you can either set this paramter to ```None``` or leave it as ```None``` is default value.

    control_flow_col_name: str = OCEL_COL_NAME_CONTROL_FLOW
        Name of the column which contains the control flow in the returned pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame:
        A DataFrame consisting of a subset of ```ocel.objecs``` that match the given object_types. Additional a column is appended with the name as defined
        in the parameter ````control_flow_col_name``` that contains the control-flow as a list of the ocel:activity column of the occured events sorted by theire timestamp.

    """
    relations = ocel.relations
    objects = ocel.objects
    if object_types != None:
        # type checking here if object_types is really a 'list'?        
        relations = relations[relations[OCEL_COL_NAME_OBJECT_TYPE].isin(object_types)]
        objects = objects[objects[OCEL_COL_NAME_OBJECT_TYPE].isin(object_types)]
    # sorting is relevant to get the events in order. the order of events that have the same timestamp is arbitrary
    relations_sorted = relations.sort_values([OCEL_COL_NAME_OBJECT_TYPE, OCEL_COL_NAME_TIMESTAMP], axis=0, ascending=True)
    # groupby statement prserves the order of the items per group.
    # documentation link: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-sorting
    # sort=False is for sorting the group keys
    cfs = relations_sorted.groupby(by=[OCEL_COL_NAME_OBJECT_TYPE, OCEL_COL_NAME_OID], sort=False)[OCEL_COL_NAME_ACTIVITY].agg(control_flow_list_attr = list)
    cfs.rename(columns={'control_flow_list_attr': control_flow_col_name}, inplace=True)

    # if an object instance is not assigned to any event, it may not occur in the dataframe cfs. thats why its a left join to relations (all relevant data)
    full_data = objects.join(cfs, on=[OCEL_COL_NAME_OBJECT_TYPE, OCEL_COL_NAME_OID], how='left')
    full_data[control_flow_col_name] = full_data[control_flow_col_name].apply(lambda d: d if isinstance(d, list) else []) # replacing NaN or non list values with empty list

    return full_data

# takes data from events. DataFrame events needs to have ocel:eid column and a column named as defined in group_col_name.
# Via this groups are created and then sub ocels returned.
def ocel_group_by_events(ocel: OCEL, events: pd.DataFrame, group_col_name: str, objects: pd.DataFrame  = None, ) -> dict[Any, OCEL]:
    """
    Groups the given object centric event log (ocel) into several ocel's. Therefore the data from the parameter events is used which is expected to contain a
    column with the name ocel:eid (unique event identifier in ```ocel.events```) and a column which contains a name of the group a event is assigned to
    (Parameter: ```group_col_name```). Its possible that a unique ocel:eid occurs multiple times in the parameter ```events```.

    The returned dictionary of ocel's identified by the group/cluster identifier of the events (data from the column ```group_col_name```) is generated as follows.
    For each group/cluster in the paramter ```events``` a new ocel is created which consists of all events that are assigned to that group/cluster.
    Then all relations and related objects from the original ocel that are connected to the events are added as well.

    UPDATE:
    Additionally all objects from the new parameter objects are added to a cluster via the same column name. This prevents the following edge-case:
    Consider a OCEL that has objects that are not connected to any event. These objects may be added to one OCEL as well. This can be defined by the user if 
    the ```objects``` parameter is not empty and has a column with the name ```group_col_name```.

    Parameters
    ---------
    ocel: OCEL
        The object-centric event log to use by the function.

    events: pandas.DataFrame
        The event data enriched by a column that defines the cluster a event is in.

    group_col_name: str
        The name of the column in the DataFrame ```èvents``` that contains the name of a cluster/group a event is in.

    objects: pandas.DataFrame
        Defines additional information to which cluster objects belong. This is only used additional by the function. Meaning the OCELs are generated and afterwards
        the objects from this table are added to clusters.

    Returns
    -------
    dict[Any, OCEL]:
        A dictionary of OCEL data that is indexed by the name of the cluster/group the ocel is representing.
    """
    EVENT_COLS = [group_col_name, OCEL_COL_NAME_EID]
    if not all(col in events.columns for col in EVENT_COLS):
        raise Exception('One mandatory column is missing. Ensure that events contains every of the following columns: "' + '", "'.join(EVENT_COLS) + '"')

    if objects is not None:
        OBJECT_COLS = [group_col_name, OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE]
        if not all(col in objects.columns for col in OBJECT_COLS):
            raise Exception('One mandatory column is missing. Ensure that objects contains every of the following columns: "' + '", "'.join(OBJECT_COLS) + '"')
        
    groups = events[group_col_name]
    if objects is not None:
        groups = np.concatenate((groups, objects[group_col_name]), axis=0)
    groups = np.unique(groups)
    groups = np.sort(groups)

    res = {}
    for group in groups:
        tmp_event_eids = events[events[group_col_name] == group][OCEL_COL_NAME_EID].unique()
        res[group] = ocel_filter_by_events(ocel, tmp_event_eids)
        
        if objects is not None:            
            # objects from the new sub-ocel
            subocel_objects = res[group].objects
            # index from current objects:
            subocel_objects_index = subocel_objects.set_index([OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE]).index
           
            # index of relevant objects which may be not present in tmp_new_ocel_objects already:
            user_objects = objects[objects[group_col_name] == group][[OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE]].set_index([OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE])
            # retrieving more data to the objects specified by the user.
            user_objects = ocel.objects.join(user_objects, on=[OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE], how="inner").set_index([OCEL_COL_NAME_OID, OCEL_COL_NAME_OBJECT_TYPE], drop=False)
            
            # Getting objects that are defined by the user for this cluster but not present in the clustered ocel already:
            new_objects = user_objects.drop(list(subocel_objects_index), errors='ignore')

            # concating new object data and setting to OCEL.
            res[group].objects = pd.concat([subocel_objects, new_objects], axis=0)

    return res

# filters the given ocel data to all nodes that contain only the given events (by ocel:eid)
def ocel_filter_by_events(ocel: OCEL, events: np.array) -> OCEL:
    """
    Filters the given ocel-data down to the events which ocel:eid's are contained in the given events array. Additional the ocel.relations and ocel.objects are also
    filtered. The result only contains data that is related to the given ocel:eid's.

    This function works also if events is empty. It returns an empty OCEL then.

    Parameters
    ----------
    ocel: OCEL
        The object-centric event log to use by the function.

    events: numpy.array
        Array of ocel:eid's which should by used to filter the original ocel-data.

    Returns
    -------
    OCEL:
        The resulting ocel-data.
    """
    # setting up dataframe for joining
    df_event_ids = pd.DataFrame(events, columns=['ocel:eid'])
    df_event_ids['ocel:eid'] = df_event_ids['ocel:eid'].astype(str)
    df_event_ids.set_index('ocel:eid', inplace=True)
    
    # creating relation data
    res_relations = ocel.relations.join(df_event_ids, on='ocel:eid', how='right') # get all relations for events, no more (right join)
    # creating object data
    res_objects = res_relations[['ocel:oid', 'ocel:type']].join(ocel.objects.set_index(['ocel:oid', 'ocel:type']), on=['ocel:oid', 'ocel:type'], how='left')[ocel.objects.columns]
    # creating event data
    res_events = ocel.events.join(df_event_ids, on=['ocel:eid'], how='right')
    # assembling ocel 
    res = OCEL(res_events, res_objects, res_relations, None, None)
    return res