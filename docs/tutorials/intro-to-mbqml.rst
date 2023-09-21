An introduction to MB-QML
=========================

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction

.. ipython:: python

   gs = mp.templates.muta(2,1, one_column=True)
   gs[3] = mp.Ment('X')
   gs[8] = mp.Ment('X')
   ps = mp.PatternSimulator(gs)
   @savefig muta_mbqml.png width=1000px
   mp.draw(gs)


.. ipython:: python
    from pathos.multiprocessing import ProcessingPool as Pool

    def loss(output, target):
        avg_fidelity = 0
        for sty, out in zip(target, output):
            sty = mp.calculator.pure2density(sty)
            avg_fidelity += 1-mp.calculator.fidelity(sty, out)
        ans = (avg_fidelity/len(target))
        return ans

    def prediction_single_state(thetas, st):
        ps.reset(input_state=st)
        statek = ps(thetas)
        return statek

    def prediction(thetas, statesx):
        thetas = np.copy(thetas)
        pool = Pool()
        output = pool.map(prediction_single_state, [thetas]*len(statesx), statesx)
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)



.. ipython:: python
    runs_train = {}
    runs_test = {}

    NUM_STEPS = 100
    NUM_RUNS = 20

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

.. ipython:: python
    