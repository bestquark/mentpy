Fisher information classifier
=============================


.. meta::
   :description: MB-QML to classify quantum states using Fisher information.
   :keywords: mb-qml, mbqc, measurement-based quantum machine learning, qml

**Author(s):** `Luis Mantilla <https://x.com/realmantilla>`_, Polina Feldmann, Dmytro Bondarenko

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction

Here, we will learn now to perform a classification of quantum states according to their Fisher information. First, we generate a dataset of states 

.. ipython:: python
      
   def gen_states_polina(numstates):
      st1a = np.array([1,0,0,0])
      st1b = np.array([0,0,0,1]) # must be orthogonal
      randtheta1 = np.pi / 2 * np.random.rand(numstates)
      randphi1 = 2 * np.pi * np.random.rand(numstates)

      states1 = np.zeros((numstates, 4), dtype='complex')
      for k in range(numstates):
         th = randtheta1[k]
         phi = randphi1[k]
         states1[k] = np.cos(th) * st1a + np.exp(1j * phi) * np.sin(th) * st1b

      st2a = np.array([1,1,1,1]) / 2.
      st2b = np.array([1,-1,-1,1]) / 2.
      randtheta2 = np.pi / 2 * np.random.rand(numstates)
      randphi2 = 2 * np.pi * np.random.rand(numstates)

      states2 = np.zeros((numstates, 4), dtype='complex')
      for k in range(numstates):
         th = randtheta2[k]
         phi = randphi2[k]
         states2[k] = np.cos(th) * st2a + np.exp(1j * phi) * np.sin(th) * st2b

      states = np.concatenate((states1, states2))
      return states

