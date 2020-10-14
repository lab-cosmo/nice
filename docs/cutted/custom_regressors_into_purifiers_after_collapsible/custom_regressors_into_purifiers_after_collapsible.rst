Our custom class looks like this:

.. code:: ipython3

    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import Ridge
    
    class AdaptiveRidge:
        def __init__(self):
            pass
        
        def fit(self, X, y):
            minimum = None
            self.best_alpha_ = None
            for alpha in np.logspace(-25, 10, 300):
                regressor = Ridge(alpha = alpha, fit_intercept = False)
                predictions = cross_val_predict(regressor, X, y)            
                now = np.mean((predictions - y) ** 2)
                if (minimum is None) or (now < minimum):
                    minimum = now
                    self.best_alpha_ = alpha
                
            self.ridge_ = Ridge(alpha = self.best_alpha_, fit_intercept = False)
            self.ridge_.fit(X, y)
            
        def predict(self, X):
            return self.ridge_.predict(X)
        
        def get_params(self, deep=True):        
            return {}
    
        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self

During fitting it estimates best value of regularization by cross
validation using training data. There are additional methods get\_params
and set\_params. These methods are required for sklearn.base.clone
function. More details about it
`here <https://scikit-learn.org/stable/developers/develop.html>`__ (It
is necessary to read only cloning section).

Let's use it:

.. code:: ipython3

    from scipy.linalg import LinAlgWarning
    
    nice = StandardSequence([StandardBlock(ThresholdExpansioner(50),
                                                 None,
                                                 IndividualLambdaPCAsBoth(20),
                                                 ThresholdExpansioner(50, mode = 'invariants'),
                                                 None,
                                                 None),
                                    StandardBlock(ThresholdExpansioner(50),
                                                 CovariantsPurifierBoth(regressor = AdaptiveRidge(),
                                                                        max_take = 20),
                                                 IndividualLambdaPCAsBoth(10),
                                                 ThresholdExpansioner(50, mode = 'invariants'),
                                                 InvariantsPurifier(regressor = AdaptiveRidge(),
                                                                        max_take = 20),
                                                 InvariantsPCA(20)),
                                    ])
    
    
    with warnings.catch_warnings():
        # a lot of ill conditioned matrices with super small alpha 
        warnings.filterwarnings("ignore", category= LinAlgWarning)  
        nice.fit(coefficients)
        
    
    res = nice.transform(coefficients)

It is possible to access best alpha parameters for all paritiies and
lambda chanels in the final model:

(convenient getters might be added in the next version of NICE)

.. code:: ipython3

    for lambd in range(6):
        if (nice.blocks_[1].covariants_purifier_.
            even_purifier_.purifiers_[lambd]):
            print("parity: even; lambda: {}; best alpha: {}".format(lambd, nice.blocks_[1].covariants_purifier_.
                  even_purifier_.purifiers_[lambd].regressor_.best_alpha_))
        if (nice.blocks_[1].covariants_purifier_.
            odd_purifier_.purifiers_[lambd]):
            print("parity odd; lambda: {}; best alpha: {}".format(lambd, nice.blocks_[1].covariants_purifier_.
                  odd_purifier_.purifiers_[lambd].regressor_.best_alpha_))


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        parity: even; lambda: 0; best alpha: 1.5996073018614912e-19
        parity: even; lambda: 1; best alpha: 3.1744774091092e-20
        parity odd; lambda: 1; best alpha: 2.0944511431514688e-19
        parity: even; lambda: 2; best alpha: 3.1744774091092e-20
        parity odd; lambda: 2; best alpha: 1e-25
        parity: even; lambda: 3; best alpha: 2.4244620170823406e-20
        parity odd; lambda: 3; best alpha: 2.7423765732649412e-19
        parity: even; lambda: 4; best alpha: 2.4244620170823406e-20
        parity odd; lambda: 4; best alpha: 1.2216773489967981e-19
        parity: even; lambda: 5; best alpha: 1e-25
        parity odd; lambda: 5; best alpha: 1e-25
    </p>
    </pre>
    </embed>
    
The same for InvariantsPurifier:

.. code:: ipython3

    print("best alpha of invariants purifier: ", 
          nice.blocks_[1].invariants_purifier_.regressor_.best_alpha_)


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        best alpha of invariants purifier:  1.381873305653628e-18
    </p>
    </pre>
    </embed>
    
