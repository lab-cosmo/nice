Now we need to call **.transform** method with **return_only_invariants
= False**, which is the default value:

.. code:: ipython3

    data_even, data_odd, invariants_even = nice[1].transform(train_coefficients[1])

Result is **data_even**, **data_odd** and **invariants_even**. The first
two objects are covariants. The last one is invariants.

There is another important symmetry in addition to the translational and
rotational one. Usually, atomic properties, such as energy, also
transform in a certain way with respect to inversion. Particularly,
energy is invariant with respect to it.

In NICE, features are separated into two groups - the ones which are
invariant with respect to inversion and the ones that change their sign.
The first ones are called even; the second ones are called odd.

Now let’s take a look at the returned objects more closely:

**Invariants** is the same object as in the previous tutorial -
dictionary, where keys are body order.

.. code:: ipython3

    for key in invariants_even.keys():
        print(invariants_even[key].shape)

Returned covariants are covariants after the last block, i. e. in our
case of body order 4. (functionality to get all covariants of all body
order from **StandardSequence** will be added in the next version of
NICE)

Even covariants are packed in the class Data, which has two relevant
fields - **.covariants\_** and **.actual_sizes\_**. (getters are also to
be added in the next version) First is np.array with covariants
themselves. It has following indexing -**[environmental_index,
feature_index, lambda, m]**. But the problem is that for each lambda
channel, the actual number of features is different. Thus, the shape of
this array doesn’t reflect the real number of meaningful entries.
Information about the actual number of features is stored in
**.actual_sizes\_**:

.. code:: ipython3

    print(type(data_even))
    print("shape of even covariants array: {}".format(data_even.covariants_.shape))
    print("actual sizes of even covariants: {}".format(data_even.actual_sizes_))

It is the same for odd covariants:

.. code:: ipython3

    print("shape of odd covariants array: {}".format(data_odd.covariants_.shape))
    print("actual sizes of odd covariants: {}".format(data_odd.actual_sizes_))

There is one other point - for each lambda channel the size of covariant
vectors is (2 \* lambda + 1). These vectors are stored from the
beginning. It means that the meaningful entries for each lambda are
located in **[:, :, lambda, :(2 \* lambda + 1)]**

In the `nice
article <https://aip.scitation.org/doi/10.1063/5.0021116>`__ another
definition of **parity** is used. Covariants are split into **true** and
**pseudo** groups. All the covariants in the **true** group are
transformed with respect to inversion as (-1)^lambda, while all the
covariants in the **pseudo** group are transformed as (-1) ^ (lambda +
1).

There is a special class - **ParityDefinitionChanger** to switch between
these definitions:

.. code:: ipython3

    data_true, data_pseudo = ParityDefinitionChanger().transform(
        data_even, data_odd)
    
    print(data_true.covariants_.shape)
    print(data_true.actual_sizes_)
    
    print(data_pseudo.covariants_.shape)
    print(data_pseudo.actual_sizes_)

Since this transformation is symmetric, we can use this once again to go
back from the true and pseudo covariants to even and odd:

.. code:: ipython3

    data_even, data_odd = ParityDefinitionChanger().transform(
        data_true, data_pseudo)

There is one other discrepancy - covariants defined in the nice article,
are smaller by the factor of (2 \* lambda + 1). Thus, the last step to
get full compliance is the following:

.. code:: ipython3

    for lambd in range(6):
        data_true.covariants_[:, :data_true.actual_sizes_[lambd],
                              lambd, :(2 * lambd + 1)] /= (2 * lambd + 1)
        data_pseudo.covariants_[:, :data_pseudo.actual_sizes_[lambd],
                                lambd, :(2 * lambd + 1)] /= (2 * lambd + 1)
