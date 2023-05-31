import clusteval.dbindex as dbindex
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
from clusteval.plot_dendrogram import plot_dendrogram
import pypickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples, silhouette_score

from tqdm import tqdm
# from cuml import DBSCAN
import wget
import os
from sklearn.metrics import silhouette_score

# %% Class
class clusteval:
    """clusteval - Cluster evaluation."""

    def __init__(self, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, savemem=False, verbose=1, params_dbscan={'eps':None, 'epsres':50, 'min_samples':0.01, 'norm':False, 'n_jobs':-1}):
        """Initialize clusteval with user-defined parameters.

        Description
        -----------
        clusteval is a python package that provides various evaluation approaches to measure the goodness of the unsupervised clustering.

        Parameters
        ----------
        cluster : str, (default: 'agglomerative')
            Type of clustering.
                * 'agglomerative'
                * 'kmeans'
                * 'dbscan'
                * 'hdbscan'
                * 'optics' # TODO
        evaluate : str, (default: 'silhouette')
            Evaluation method for cluster validation.
                * 'silhouette'
                * 'dbindex'
                * 'derivative'
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
                * 'ward'
                * 'single'
                * 'complete'
                * 'average'
                * 'weighted'
                * 'centroid'
                * 'median'
        min_clust : int, (default: 2)
            Number of clusters that is evaluated greater or equals to min_clust.
        max_clust : int, (default: 25)
            Number of clusters that is evaluated smaller or equals to max_clust.
        savemem : bool, (default: False)
            Save memmory when working with large datasets. Note that htis option only in case of KMeans.
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        dict : The output is a dictionary containing the following keys:

        Examples
        --------
        >>> # Import library
        >>> from clusteval import clusteval
        >>> # Initialize clusteval with default parameters
        >>> ce = clusteval()
        >>>
        >>> # Generate random data
        >>> from sklearn.datasets import make_blobs
        >>> X, labels_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
        >>>
        >>> # Fit best clusters
        >>> results = ce.fit(X)
        >>>
        >>> # Make plot
        >>> ce.plot()
        >>>
        >>> # Scatter plot
        >>> ce.scatter(X)
        >>>
        >>> # Dendrogram
        >>> ce.dendrogram()

        """
        if ((min_clust is None) or (min_clust<2)):
            min_clust=2
        if ((max_clust is None) or (max_clust<min_clust)):
            max_clust=min_clust + 1

        if not np.any(np.isin(evaluate, ['silhouette', 'dbindex', 'derivative'])): raise ValueError("evaluate has incorrect input argument [%s]." %(evaluate))
        if not np.any(np.isin(cluster, ['agglomerative', 'kmeans', 'dbscan', 'hdbscan'])): raise ValueError("cluster has incorrect input argument [%s]." %(cluster))

        # Set parameters for dbscan
        dbscan_defaults = {'metric': metric, 'min_clust': min_clust, 'max_clust': max_clust, 'eps': None, 'epsres': 50, 'min_samples': 0.01, 'norm': False, 'n_jobs': -1, 'verbose': verbose}
        params_dbscan = {**dbscan_defaults, **params_dbscan}
        self.params_dbscan = params_dbscan

        # Store in object
        self.evaluate = evaluate
        self.cluster = cluster
        self.metric = metric
        self.linkage = linkage
        self.min_clust = min_clust
        self.max_clust = max_clust
        self.savemem = savemem
        self.verbose = verbose

    # Fit
    def fit(self, X):
        """Cluster validation.

        Parameters
        ----------
        X : Numpy-array.
            The rows are the features and the colums are the samples.

        Returns
        -------
        dict. with various keys. Note that the underneath keys can change based on the used evaluation method.
            evaluate: str
                evaluate name that is used for cluster evaluation.
            score: pd.DataFrame()
                The scoring values per clusters [silhouette, dbindex] provide this information.
            labx: list
                Cluster labels.
            fig: list
                Relevant information to make the plot.

        """
        if 'array' not in str(type(X)): raise ValueError('Input data must be of type numpy array')
        max_d, max_d_lower, max_d_upper = None, None, None
        self.Z = []

        # Cluster using on metric/linkage
        if self.verbose>=3: print('\n[clusteval] >Fit using %s with metric: %s, and linkage: %s' %(self.cluster, self.metric, self.linkage))
        # Compute linkages
        if self.cluster!='kmeans':
            self.Z = scipy_linkage(X, method=self.linkage, metric=self.metric)

        # Choosing method
        if (self.cluster=='agglomerative') or (self.cluster=='kmeans'):
            if self.evaluate=='silhouette':
                self.results = fit(X, Z=self.Z, cluster=self.cluster, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, savemem=self.savemem, verbose=self.verbose)
            elif self.evaluate=='dbindex':
                self.results = dbindex.fit(X, Z=self.Z, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, savemem=self.savemem, verbose=self.verbose)
            elif self.evaluate=='derivative':
                self.results = derivative.fit(X, Z=self.Z, cluster=self.cluster, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, verbose=self.verbose)
        elif (self.cluster=='dbscan') and (self.evaluate=='silhouette'):
            self.results = dbscan.fit(X, eps=self.params_dbscan['eps'], epsres=self.params_dbscan['epsres'], min_samples=self.params_dbscan['min_samples'], metric=self.metric, norm=self.params_dbscan['norm'], n_jobs=self.params_dbscan['n_jobs'], min_clust=self.min_clust, max_clust=self.max_clust, verbose=self.verbose)
        elif self.cluster=='hdbscan':
            try:
                import clusteval.hdbscan as hdbscan
            except:
                raise ValueError('[clusteval] >hdbscan must be installed manually. Try to: <pip install hdbscan> or <conda install -c conda-forge hdbscan>')
            self.results = hdbscan.fit(X, min_samples=None, metric=self.metric, norm=True, n_jobs=-1, min_clust=self.min_clust, verbose=self.verbose)
        else:
            raise ValueError('[clusteval] >The combination cluster"%s", evaluate="%s" is not implemented.' %(self.cluster, self.evaluate))

        # Compute the dendrogram threshold
        max_d, max_d_lower, max_d_upper = None, None, None

        # Compute the dendrogram threshold
        if (self.cluster!='kmeans') and hasattr(self, 'results') and (self.results['labx'] is not None) and (len(np.unique(self.results['labx']))>1):
            # print(self.results['labx'])
            max_d, max_d_lower, max_d_upper = _compute_dendrogram_threshold(self.Z, self.results['labx'], verbose=self.verbose)

        if self.results['labx'] is not None:
            if self.verbose>=3: print('[clusteval] >Optimal number clusters detected: [%.0d].' %(len(np.unique(self.results['labx']))))

        self.results['max_d'] = max_d
        self.results['max_d_lower'] = max_d_lower
        self.results['max_d_upper'] = max_d_upper
        if self.verbose>=3: print('[clusteval] >Fin.')

        # Return
        return self.results

    # Plot
    def plot(self, title=None, figsize=(15, 8), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, verbose=3):
        """Make a plot.

        Parameters
        ----------
        figsize : tuple, (default: (15, 8).
            Size of the figure (height,width).
        savefig : dict.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            {'dpi':'figure',
            'format':None,
            'metadata':None,
            'bbox_inches': None,
            'pad_inches':0.1,
            'facecolor':'auto',
            'edgecolor':'auto',
            'backend':None}
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        tuple: (fig, ax)

        """
        fig, ax = None, None
        if (self.results is None) or (self.results['labx'] is None):
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function first.')
            return None

        if (self.cluster=='agglomerative') or (self.cluster=='kmeans'):
            if self.evaluate=='silhouette':
                fig, ax = plot(self.results, figsize=figsize)
            elif self.evaluate=='dbindex':
                fig, ax = dbindex.plot(self.results, figsize=figsize)
            elif self.evaluate=='derivative':
                fig, ax = derivative.plot(self.results, title=title, figsize=figsize)
        elif self.cluster=='dbscan':
            fig, ax = dbscan.plot(self.results, figsize=figsize)
        elif self.cluster=='hdbscan':
            import clusteval.hdbscan as hdbscan
            fig, ax = hdbscan.plot(self.results, figsize=figsize, savefig=savefig)
        # Save figure
        if (savefig['fname'] is not None) and (fig is not None) and (self.cluster!='hdbscan'):
            if verbose>=3: print('[clusteval] >Saving plot: [%s]' %(savefig['fname']))
            fig.savefig(**savefig)

        # Return
        return fig, ax

    # Plot
    def scatter(self, X, dot_size=75, figsize=(15, 8), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, verbose=3):
        """Make a plot.

        Parameters
        ----------
        X : array-like, (default: None)
            Input dataset used in the .fit() funciton. Some plots will be more extensive if the input data is also provided.
        dot_size : int, (default: 50)
            Size of the dot in the scatterplot
        figsize : tuple, (default: (15,8).
            Size of the figure (height,width).
        savefig : dict.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            {'dpi':'figure',
            'format':None,
            'metadata':None,
            'bbox_inches': None,
            'pad_inches':0.1,
            'facecolor':'auto',
            'edgecolor':'auto',
            'backend':None}
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        None.

        """
        if (self.results is None) or (self.results['labx'] is None):
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function first.')
            return None
        # Make scatterplot
        fig, ax1, ax2 = scatter(self.results, X=X, dot_size=dot_size, figsize=figsize, savefig=savefig)
        # Return
        return (fig, ax1, ax2)

    # Plot dendrogram
    def dendrogram(self, X=None, labels=None, leaf_rotation=90, leaf_font_size=12, orientation='top', show_contracted=True, max_d=None, showfig=True, metric=None, linkage=None, truncate_mode=None, figsize=(15, 10), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, verbose=3):
        """Plot Dendrogram.

        Parameters
        ----------
        X : numpy-array (default : None)
            Input data.
        labels : list, (default: None)
            Plot the labels. When None: the index of the original observation is used to label the leaf nodes.
        leaf_rotation : int, (default: 90)
            Rotation of the labels [0-360].
        leaf_font_size : int, (default: 12)
            Font size labels.
        orientation : string, (default: 'top')
            Direction of the dendrogram: 'top', 'bottom', 'left' or 'right'
        show_contracted : bool, (default: True)
            The heights of non-singleton nodes contracted into a leaf node are plotted as crosses along the link connecting that leaf node.
        max_d : Float, (default: None)
            Height of the dendrogram to make a horizontal cut-off line.
        showfig : bool, (default = True)
            Plot the dendrogram.
        metric : str, (default: 'euclidean').
            Distance measure for the clustering, such as 'euclidean','hamming', etc.
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
            'ward','single',',complete','average','weighted','centroid','median'.
        truncate_mode : string, (default: None)
            Truncation is used to condense the dendrogram, which can be based on: 'level', 'lastp' or None
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).
        savefig : dict.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            {'dpi':'figure',
            'format':None,
            'metadata':None,
            'bbox_inches': None,
            'pad_inches':0.1,
            'facecolor':'auto',
            'edgecolor':'auto',
            'backend':None}
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        results : dict
            * labx : int : Cluster labels based on the input-ordering.
            * order_rows : string : Order of the cluster labels as presented in the dendrogram (left-to-right).
            * max_d : float : maximum distance to set the horizontal threshold line.
            * max_d_lower : float : maximum distance lowebound
            * max_d_upper : float : maximum distance upperbound

        """
        fig, ax = None, None
        if (self.results is None) or (self.results['labx'] is None):
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function first.')
            return None

        # Set parameters
        no_plot = False if showfig else True
        max_d_lower, max_d_upper = None, None

        # Check whether
        if (metric is not None) and (linkage is not None) and (X is not None):
            if self.verbose>=2: print('[clusteval] >Compute dendrogram using metric=%s, linkage=%s' %(metric, linkage))
            Z = scipy_linkage(X, method=linkage, metric=metric)
        elif (metric is not None) and (linkage is not None) and (X is None):
            if self.verbose>=2: print('[clusteval] >To compute the dendrogram, also provide the data: X=data <return>')
            return None
        elif (not hasattr(self, 'Z')):
            # Return if Z is not computed.
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function (no kmeans) <return>')
            return None
        else:
            # if self.verbose>=3: print('[clusteval] >Plotting the dendrogram with optimized settings: metric=%s, linkage=%s, max_d=%.3f. Be patient now..' %(self.metric, self.linkage, self.results['max_d']))
            Z = self.Z
            metric = self.metric
            linkage = self.linkage

        if self.cluster=='kmeans':
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function with metric that is different than kmeans <return>')
            return None

        if max_d is None:
            max_d = self.results['max_d']
            max_d_lower = self.results['max_d_lower']
            max_d_upper = self.results['max_d_upper']

        # Make the dendrogram
        if showfig:
            fig, ax = plt.subplots(figsize=figsize)
        annotate_above = max_d
        results = plot_dendrogram(Z, labels=labels, leaf_rotation=leaf_rotation, leaf_font_size=leaf_font_size, orientation=orientation, show_contracted=show_contracted, annotate_above=annotate_above, max_d=max_d, truncate_mode=truncate_mode, ax=ax, no_plot=no_plot)

        # Compute cluster labels
        if self.verbose>=3: print('[clusteval] >Compute cluster labels.')
        labx = fcluster(Z, max_d, criterion='distance')

        # Store results
        results['order_rows'] = np.array(results['ivl'])
        results['labx'] = labx
        results['max_d'] = max_d
        results['max_d_lower'] = max_d_lower
        results['max_d_upper'] = max_d_upper
        results['ax'] = ax

        # Save figure
        if (savefig['fname'] is not None) and (fig is not None):
            if verbose>=3: print('[clusteval] >Saving dendrogram: [%s]' %(savefig['fname']))
            fig.savefig(**savefig)

        return results

    def save(self, filepath='clusteval.pkl', overwrite=False):
        """Save model in pickle file.

        Parameters
        ----------
        filepath : str, (default: 'clusteval.pkl')
            Pathname to store pickle files.
        overwrite : bool, (default=False)
            Overwite file if exists.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.

        Returns
        -------
        bool : [True, False]
            Status whether the file is saved.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'clusteval.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=3)
        # return
        return status

    def load(self, filepath='clusteval.pkl', verbose=3):
        """Restore previous results.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'clusteval.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        # Restore the data in self
        if storedata is not None:
            self.results = storedata['results']
            return self.results

    def import_example(self, data='titanic', url=None, sep=',', verbose=3):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.
        verbose : int, (default: 3)
            Print message to screen.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep, verbose=verbose)


# %% Compute dendrogram threshold
def _compute_dendrogram_threshold(Z, labx, verbose=3):
    if verbose>=3: print('[clusteval] >Compute dendrogram threshold.')
    Iloc = np.isin(Z[:, 3], np.unique(labx, return_counts=True)[1])
    max_d_lower = np.max(Z[Iloc, 2])
    # Find the next level
    if np.any(Z[:, 2] > max_d_lower):
        max_d_upper = Z[np.where(Z[:, 2] > max_d_lower)[0][0], 2]
    else:
        max_d_upper = np.sort(Z[Iloc, 2])[-2]
    # Average the max_d between the start and stop level
    max_d = max_d_lower + ((max_d_upper - max_d_lower) / 2)
    # Return
    return max_d, max_d_lower, max_d_upper


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
    verbose : int, (default: 3)
        Print message to screen.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
            sep=';'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('[clusteval] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[clusteval] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[clusteval] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df

def fit(X, cluster='agglomerative', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, Z=None, savemem=False, verbose=3):
    """This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering evaluate.

    Parameters
    ----------
    X : Numpy-array,
        Where rows is features and colums are samples.
    cluster : str, (default: 'agglomerative')
        Clustering evaluation type for clustering.
            * 'agglomerative'
            * 'kmeans'
    metric : str, (default: 'euclidean').
        Distance measure for the clustering, such as 'euclidean','hamming', etc.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    min_clust : int, (default: 2)
        Number of clusters that is evaluated greater or equals to min_clust.
    max_clust : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to max_clust.
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that htis option only in case of KMeans.
    Z : Object, (default: None).
        This will speed-up computation if you readily have Z. e.g., Z=linkage(X, method='ward', metric='euclidean').
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information is returned.

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used evaluatetype.
    evaluate: str
        evaluate name that is used for cluster evaluation.
    score: pd.DataFrame()
        The scoring values per clusters.
    labx: list
        Cluster labels.
    fig: list
        Relevant information to make the plot.

    Examples
    --------
    >>> # Import library
    >>> import clusteval.silhouette as silhouette
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> # Example 1:
    >>> Generate demo data
    >>> X, labels_true = make_blobs(n_samples=750, centers=5, n_features=10)
    >>> # Fit with default parameters
    >>> results = silhouette.fit(X)
    >>> # plot
    >>> silhouette.scatter(results, X)
    >>> silhouette.plot(results)
    >>>
    >>> # Example 2:
    >>> # Try also alternative dataset
    >>> X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
    >>> # Fit with some specified parameters
    >>> results = silhouette.fit(X, metric='kmeans', savemem=True)
    >>> # plot
    >>> silhouette.scatter(results, X)
    >>> silhouette.plot(results)

    References
    ----------
    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    """
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose'] = verbose
    Param['cluster'] = cluster
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['min_clust'] = min_clust
    Param['max_clust'] = max_clust
    Param['savemem'] = savemem
    if verbose>=3: print('[clusteval] >Evaluate using silhouette.')

    # Savemem for kmeans
    if Param['cluster']=='kmeans':
        if Param['savemem']:
            kmeansmodel = MiniBatchKMeans
            if Param['verbose']>=3: print('[clusteval] >Save memory enabled for kmeans with evaluation silhouette.')
        else:
            kmeansmodel = KMeans

    # Cluster hierarchical using on metric/linkage
    if (Z is None) and (Param['cluster']!='kmeans'):
        Z = scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])

    # Setup storing parameters
    clustcutt = np.arange(Param['min_clust'], Param['max_clust'])
    silscores = np.zeros((len(clustcutt))) * np.nan
    sillclust = np.zeros((len(clustcutt))) * np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt))):
        # Cut the dendrogram for i clusters
        if Param['cluster']=='kmeans':
            labx = kmeansmodel(n_clusters=clustcutt[i], verbose=0).fit(X).labels_
        else:
            labx = fcluster(Z, clustcutt[i], criterion='maxclust')

        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        sillclust[i] = len(np.unique(labx))
        # Compute silhouette (can only be done if more then 1 cluster)
        if sillclust[i]>1:
            silscores[i] = silhouette_score(X, labx)

    # Convert to array
    clustlabx = np.array(clustlabx)

    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(silscores)==False
    I2 = sillclust>=Param['min_clust']
    I3 = sillclust<=Param['max_clust']
    Iloc = I1 & I2 & I3

    if verbose>=5:
        print(clustlabx)
        print('Iloc: %s' %(str(Iloc)))
        print('silscores: %s' %(str(silscores)))
        print('sillclust: %s' %(str(sillclust)))
        print('clustlabx: %s' %(str(clustlabx)))

    if sum(Iloc)>0:
        # Get only clusters of interest
        silscores = silscores[Iloc]
        sillclust = sillclust[Iloc]
        clustlabx = clustlabx[Iloc, :]
        clustcutt = clustcutt[Iloc]
        idx = np.argmax(silscores)
        clustlabx = clustlabx[idx, :] - 1
    else:
        if verbose>=3: print('[clusteval] >No clusters detected.')
        if len(clustlabx.shape)>1:
            clustlabx = np.zeros(clustlabx.shape[1]).astype(int)
        else:
            clustlabx = [0]

    # Store results
    results = {}
    results['evaluate']='silhouette'
    results['score'] = pd.DataFrame(np.array([sillclust, silscores]).T, columns=['clusters', 'score'])
    results['score']['clusters'] = results['score']['clusters'].astype(int)
    results['labx'] = clustlabx
    results['fig'] = {}
    results['fig']['silscores'] = silscores
    results['fig']['sillclust'] = sillclust
    results['fig']['clustcutt'] = clustcutt

    # Return
    return(results)

# %% plot
def plot(results, title='Silhouette vs. nr.clusters', figsize=(15, 8), ax=None, visible=True):
    """Make plot for the gridsearch over the number of clusters.

    Parameters
    ----------
    results : dict.
        Dictionary that is the output of the .fit() function.
    figsize : tuple, (default: (15,8))
        Figure size, (heigh,width).

    Returns
    -------
    tuple, (fig, ax)
        Figure and axis of the figure.

    """
    fig=None
    idx = np.argmax(results['fig']['silscores'])
    # Make figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Plot
    ax.plot(results['fig']['sillclust'], results['fig']['silscores'], color='k')
    # Plot optimal cut
    ax.axvline(x=results['fig']['clustcutt'][idx], ymin=0, ymax=results['fig']['sillclust'][idx], linewidth=2, color='r', linestyle="--")
    # Set fontsizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('xtick', labelsize=10)  # fontsize of the axes title
    plt.rc('ytick', labelsize=10)  # fontsize of the axes title
    plt.rc('font', size=10)
    # Set labels
    ax.set_xticks(results['fig']['clustcutt'])
    ax.set_xlabel('#Clusters')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(color='grey', linestyle='--', linewidth=0.2)
    # if visible:
    #     plt.show()
    # Return
    return(fig, ax)


# %% Scatter data
def scatter(y, X=None, dot_size=50, figsize=(15, 8), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, verbose=3):
    """Make scatter for the cluster labels with the samples.

    Parameters
    ----------
    y: list
        Cluster labels for the samples in X (some order).
    X : Numpy-array,
        Where rows is features and colums are samples. The first two columns of the matrix are used for plotting. Note that it is also possible provide tSNE coordinates for a better representation of the data.
    dot_size : int, (default: 50)
        Size of the dot in the scatterplot
    savefig : dict.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        {'dpi':'figure',
        'format':None,
        'metadata':None,
        'bbox_inches': None,
        'pad_inches':0.1,
        'facecolor':'auto',
        'edgecolor':'auto',
        'backend':None}
    figsize : tuple, (default: (15,8))
        Figure size, (heigh,width).

    Returns
    -------
    tuple, (fig, ax1, ax2)
        Figure and axis of the figure.

    """
    fig, ax1, ax2 = None, None, None
    if X is None:
        if verbose>=2: print('[clusteval] >Warning: Input data X is required for the scatterplot.')
        return None

    # Extract label from dict
    if isinstance(y, dict):
        y = y.get('labx', None)
    # Check y
    if (y is None) or (len(np.unique(y))==1):
        if verbose>=3: print('[clusteval] >Error: No valid labels provided.')
        return None

    # Plot silhouette samples plot
    # n_clusters = len(np.unique(y))
    n_clusters = len(set(y))
    silhouette_avg = silhouette_score(X, y)
    if verbose>=3: print('[clusteval] >Estimated number of n_clusters: %d, average silhouette_score=%.3f' %(n_clusters, silhouette_avg))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])

    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    y_lower = 10
    uiclust = np.unique(y)

    # Make 1st plot
    for i in range(0, len(uiclust)):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[y == uiclust[i]]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.Set2(float(i) / n_clusters)

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        # ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=getcolors[i], edgecolor=getcolors[i], alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(uiclust[i]))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)

    # 2nd Plot showing the actual clusters formed
    color = cm.Set2(y.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=dot_size, lw=0, alpha=0.8, c=color, edgecolor='k')
    ax2.grid(color='grey', linestyle='--', linewidth=0.2)
    ax2.set_title("Estimated cluster labels")
    ax2.set_xlabel("1st feature")
    ax2.set_ylabel("2nd feature")
    # General title
    plt.suptitle(("Silhouette analysis results in n_clusters = %d" %(n_clusters)), fontsize=14, fontweight='bold')
    # plt.show()

    # Save figure
    if (savefig['fname'] is not None) and (fig is not None):
        if verbose>=3: print('[clusteval] >Saving silhouetteplot to [%s]' %(savefig['fname']))
        fig.savefig(**savefig)

    # Return
    return(fig, ax1, ax2)
