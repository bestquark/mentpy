Parallelizing MB-QML protocols
==============================

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction


In this tutorial, we will see how to parallelize the MB-QML protocols in :mod:`mentpy`. 
Similar to the previous tutorial, we first need a MB-QML model to work with.

.. ipython:: python

   gs = mp.templates.muta(2,1, one_column=True)
   gs[3] = mp.Ment('X')
   gs[8] = mp.Ment('X')
   ps = mp.PatternSimulator(gs)
   @savefig muta_mbqml.png width=1000px
   mp.draw(gs)


Then, when we define a loss function, we can use the :mod:`pathos` package to parallelize the computation
of the infidelity between the target states and the output states.

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


This will significantly speed up the computation of the loss function!