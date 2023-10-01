An introduction to MB-QML
=========================

.. meta::
   :description: An introduction to measurement-based quantum machine learning
   :keywords: mb-qml, measurement-based quantum machine learning, quantum machine learning, mbqc

**Author(s):** `Luis Mantilla <https://twitter.com/realmantilla>`_

Quantum machine learning (QML) is a field that studies how to use parametrized quantum circuits to 
learn to identify patterns in quantum data. In measurement-based qunatum machine learning (MB-QML) 
[#thesis]_, we use a MBQC circuit with parametrized measurement angles to solve QML problems. 

In :mod:`mentpy`, MB-QML models are defined using the :class:`MBQCircuit` class. We can define a model from scratch
or use one of the templates provided in :mod:`mentpy.templates`. Here, we use the MuTA template with two 
input qubits, and fix two of the parameters to be fixed (qubits 3 and 8).

.. ipython:: python

    import numpy as np
    import mentpy as mp

    gs = mp.templates.muta(2,1, one_column=True)
    gs[3] = mp.Ment('X')
    gs[8] = mp.Ment('X')
    ps = mp.PatternSimulator(gs)
    @savefig muta_mbqml.png width=1000px
    mp.draw(gs)

To optimize the parameters of the model, we need to define a loss function. Here, we will use the 
average infidelity between the target states and the output states of the model.

.. ipython:: python

    def loss(output, target):
        avg_fidelity = 0
        for sty, out in zip(target, output):
            sty = mp.calculator.pure2density(sty)
            avg_fidelity += 1-mp.calculator.fidelity(sty, out)
        ans = (avg_fidelity/len(target))
        return ans

    def prediction_single_state(thetas, st):
        ps.reset(input_state=st)
        st = ps(thetas)
        return st

    def prediction(thetas, statesx):
        output = [prediction_single_state(thetas, st) for st in statesx]
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)

Be aware that the loss function is a global operation, which can induce barren plateaus. However,
we will ignore this issue for now. Having defined a model and a loss function, 
we can now use some data to train our model. We will use the :func:`generate_random_dataset` function 
to generate a random dataset of states :math:`\left\{(\rho_i, \sigma_i)_i \right\}_i^{N}`
where the input and target states are related by a given unitary :math:`\sigma_i = U \rho_i U^\dagger`.

.. code-block:: python

    runs_train = {}
    runs_test = {}

    NUM_STEPS = 100
    NUM_RUNS = 10

    for i in range(NUM_RUNS):
        random_gate = np.kron(mp.utils.random_special_unitary(1), np.eye(2))
        (x_train, y_train), (x_test, y_test) = mp.utils.generate_random_dataset(random_gate, 10, test_size = 0.3)

        cost_train, cost_test = [], []

        def callback(params, iter):
            cost_train.append(cost(params, x_train, y_train))
            cost_test.append(cost(params, x_test, y_test))
        
        theta = np.random.rand(len(gs.trainable_nodes))
        opt = mp.optimizers.AdamOptimizer(step_size=0.08)
        theta = opt.optimize(lambda params: cost(params, x_train, y_train), theta, num_iters=NUM_STEPS, callback=callback)
        post_cost = cost(theta, x_test, y_test)

        runs_train[i] = cost_train
        runs_test[i] = cost_test

Finally, we can average over the runs and plot the results!

References
----------

.. [#thesis] Mantilla Calderón, L. C. (2023). Measurement-based quantum machine learning (T). University of British Columbia. 