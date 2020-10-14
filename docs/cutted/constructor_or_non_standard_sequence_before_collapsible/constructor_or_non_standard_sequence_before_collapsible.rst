Constructor or non standard sequence
------------------------------------

In previous tutorials it was discussed how to perform calculations with
standard NICE scheme, which is reflected by class StandardSequence. But
NICE toolbox provides broader opportunities. It is possible, for
example, to combine latest covariants with each other at each step in
order to get 2^n body order features after n iterations.

In previous tutorials model was defined by StandardSequence class, whose
initialization method accepts instances of other classes as
ThresholdExpansioner or InvariantsPurifier. These blocks can be used by
their own to construct custom model.

First of all we need to calculate spherical expansion coefficients as in
previous tutorials:
