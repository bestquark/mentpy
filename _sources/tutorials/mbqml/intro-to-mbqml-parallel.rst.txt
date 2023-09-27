Parallelizing MB-QML protocols
==============================

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction


In measurement-based quantum machine learning, we first need to define a model. The model will 
be a MBQC circuit with parametrized measurement angles. Let's define a model using the MuTA 
ansatz with two input qubits:

.. ipython:: python

   gs = mp.templates.muta(2,1, one_column=True)
   gs[3] = mp.Ment('X')
   gs[8] = mp.Ment('X')
   ps = mp.PatternSimulator(gs)
   @savefig muta_mbqml.png width=1000px
   mp.draw(gs)


Great, now we need to define a loss function. In our case, we will use the average infidelity between
the target states and the output states. 

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

