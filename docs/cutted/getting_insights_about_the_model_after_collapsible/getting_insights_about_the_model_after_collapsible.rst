As was discussed in the first tutorial, **ThresholdExpansioner** sorts
all pairs of inputs by their pairwise importances and, after that,
produces the output only for a fixed number of the most important pairs.
This number is controlled by **num_expand**.

However, there are two reasons why the real number of covariants after
**ThresholdEpansioner** might be different from the specified one. 1)
Some pairs of input covariants do not produce features to all lambda
channels. Particularly, pair of input covariants with some l1 and l2
produces covariants only to lambda channels where \|l1 - l2\| <= lambda
<= l1 + l2. Thus, the real number of features after
**ThresholdExpanioner** would be smaller than the specified one in
**num_expand**.

2) Pairwise importances can have a lot of collisions. For instance, it
   is impossible to select such a threshold to filter out exactly 3
   pairs from the set of pairs with the following importances [1, 1, 2,
   2]. It is possible to filter out either 0, either 2, either 4, but
   not exactly 3.

Thus, it is a good idea to have the possibility to look at the actual
amount of intermediate features.

**StandardSequence** has a method **get_intermediat_shapes()**. It
returns intermediate shapes in the form of nested dictionary:

.. code:: ipython3

    intermediate_shapes = nice[1].get_intermediate_shapes()
    
    for key in intermediate_shapes.keys():
        print(key, ':', intermediate_shapes[key], end='\n\n\n')

Spectrums of pcas can be accessed in the following way: (convenient
getters will be inserted in the next version of NICE)

.. code:: ipython3

    def proper_log_plot(array, *args, **kwargs):
        '''avoiding log(0)'''
        plt.plot(np.arange(len(array)) + 1, array, *args, **kwargs)
        plt.ylim([1e-3, 1e0])
    
    
    colors = ['r', 'g', 'b', 'orange', 'yellow', 'purple']
    
    print("nu: ", 1)
    for i in range(6):  # loop over lambda channels
        if (nice[6].initial_pca_ is not None):
            if (nice[6].initial_pca_.even_pca_.pcas_[i] is not None):
                proper_log_plot(
                    nice[6].initial_pca_.even_pca_.pcas_[i].importances_,
                    color=colors[i],
                    label="lambda = {}".format(i))
    
    for i in range(6):  # loop over lambda channels
        if (nice[6].initial_pca_ is not None):
            if (nice[6].initial_pca_.odd_pca_.pcas_[i] is not None):
                proper_log_plot(
                    nice[6].initial_pca_.odd_pca_.pcas_[i].importances_,
                    '--',
                    color=colors[i],
                    label="lambda = {}".format(i))
    
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    for nu in range(len(nice[6].blocks_)):  # loop over body orders
        print("nu: ", nu + 2)
        for i in range(6):  # loop over lambda channels
            if (nice[6].blocks_[nu].covariants_pca_ is not None):
                if (nice[6].blocks_[nu].covariants_pca_.even_pca_.pcas_[i]
                        is not None):
                    proper_log_plot(nice[6].blocks_[nu].covariants_pca_.even_pca_.
                                    pcas_[i].importances_,
                                    color=colors[i],
                                    label="lambda = {}".format(i))
    
        for i in range(6):  # loop over lambda channels
            if (nice[6].blocks_[nu].covariants_pca_ is not None):
                if (nice[6].blocks_[nu].covariants_pca_.odd_pca_.pcas_[i]
                        is not None):
                    proper_log_plot(nice[6].blocks_[nu].covariants_pca_.odd_pca_.
                                    pcas_[i].importances_,
                                    '--',
                                    color=colors[i])
    
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()

(checks if pca instance is **None** are needed since it would be
**None** if the number of features for corresponding lambda channel
would be zero after the expansion step)

Inner class for single Lambda channel inherits from
sklearn.decomposition.TruncatedSVD (PCA without centering the data,
which would break covariant transformation). Thus, in addition to
**.importances\_**, **.explained_variance\_** and
**.explained_variance_ratio\_** are also accessible.

**importances\_** (which are used by subsequent
**TresholdExpansioners**) are **explained_variance\_** normalized not to
variance of input as **explained_variance_ratio\_**, but to variance of
output:

.. code:: ipython3

    print(np.sum(nice[6].blocks_[1].\
                 covariants_pca_.even_pca_.pcas_[2].explained_variance_))
    print(np.sum(nice[6].blocks_[1].\
                 covariants_pca_.even_pca_.pcas_[2].explained_variance_ratio_))
    print(np.sum(nice[6].blocks_[1].\
                 covariants_pca_.even_pca_.pcas_[2].importances_))
