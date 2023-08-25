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