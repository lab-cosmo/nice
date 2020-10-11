.. code:: ipython3

    intermediate_shapes = transformers[1].get_intermediate_shapes()
    
    for key in intermediate_shapes.keys():
        print(key, ':',  intermediate_shapes[key], end = '\n\n\n')


.. parsed-literal::

    after initial transformer : [[10, 0, 10, 0, 10, 0], [0, 10, 0, 10, 0, 10]]
    
    
    after initial pca : [[10, 0, 10, 0, 10, 0], [0, 10, 0, 10, 0, 10]]
    
    
    block nu = 1 -> nu = 2 : {'after covariants expansioner': [[52, 51, 121, 103, 131, 94], [0, 92, 82, 130, 104, 114]], 'after covariants purifier': [[52, 51, 121, 103, 131, 94], [0, 92, 82, 130, 104, 114]], 'after covariants pca': [[50, 50, 50, 50, 50, 50], [0, 50, 50, 50, 50, 50]], 'after invariants expansioner': 300, 'after invariants purifier': 300, 'after invariants pca': 200}
    
    
    block nu = 2 -> nu = 3 : {'after covariants expansioner': [[33, 51, 79, 88, 87, 73], [12, 65, 76, 88, 88, 80]], 'after covariants purifier': [[33, 51, 79, 88, 87, 73], [12, 65, 76, 88, 88, 80]], 'after covariants pca': [[33, 50, 50, 50, 50, 50], [12, 50, 50, 50, 50, 50]], 'after invariants expansioner': 300, 'after invariants purifier': 300, 'after invariants pca': 200}
    
    
    block nu = 3 -> nu = 4 : {'after covariants expansioner': [[19, 49, 57, 72, 66, 62], [25, 49, 65, 62, 73, 57]], 'after covariants purifier': [[19, 49, 57, 72, 66, 62], [25, 49, 65, 62, 73, 57]], 'after covariants pca': [[19, 49, 50, 50, 50, 50], [25, 49, 50, 50, 50, 50]], 'after invariants expansioner': 300, 'after invariants purifier': 300, 'after invariants pca': 200}
    
    


.. code:: ipython3

    def proper_log_plot(array, *args, **kwargs):
        plt.plot(np.arange(len(array)) + 1, array, *args, **kwargs)
        plt.ylim([1e-3, 1e0])
    
    colors = ['r', 'g', 'b', 'orange', 'yellow', 'purple']
    
    print("nu: ", 1)
    for i in range(6):
        if (transformers[6].initial_pca_ is not None):
            if (transformers[6].initial_pca_.even_pca_.pcas_[i] is not None):
                proper_log_plot(transformers[6].initial_pca_.even_pca_.pcas_[i].importances_,
                                color = colors[i], label = "lambda = {}".format(i))
    
    
    for i in range(6):
        if (transformers[6].initial_pca_ is not None):
            if (transformers[6].initial_pca_.odd_pca_.pcas_[i] is not None):
                proper_log_plot(transformers[6].initial_pca_.odd_pca_.pcas_[i].importances_,
                                '--', color = colors[i], label = "lambda = {}".format(i))
    
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
        
    for nu in range(len(transformers[6].blocks_)):
        print("nu: ", nu + 2)
        for i in range(6):
            if (transformers[6].blocks_[nu].covariants_pca_ is not None):
                if (transformers[6].blocks_[nu].covariants_pca_.even_pca_.pcas_[i] is not None):
                    proper_log_plot(transformers[6].blocks_[nu].covariants_pca_.even_pca_.pcas_[i].importances_, color = colors[i], label = "lambda = {}".format(i))
            
        
        for i in range(6):
            if (transformers[6].blocks_[nu].covariants_pca_ is not None):
                if (transformers[6].blocks_[nu].covariants_pca_.odd_pca_.pcas_[i] is not None):
                    proper_log_plot(transformers[6].blocks_[nu].covariants_pca_.odd_pca_.pcas_[i].importances_, '--', color = colors[i])
        
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()


.. parsed-literal::

    nu:  1



.. image:: getting_insights_about_the_model_after_collapsible_files/getting_insights_about_the_model_after_collapsible_1_1.png


.. parsed-literal::

    nu:  2



.. image:: getting_insights_about_the_model_after_collapsible_files/getting_insights_about_the_model_after_collapsible_1_3.png


.. parsed-literal::

    nu:  3



.. image:: getting_insights_about_the_model_after_collapsible_files/getting_insights_about_the_model_after_collapsible_1_5.png


.. parsed-literal::

    nu:  4



.. image:: getting_insights_about_the_model_after_collapsible_files/getting_insights_about_the_model_after_collapsible_1_7.png

