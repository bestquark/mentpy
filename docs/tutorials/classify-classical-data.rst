Classifying classical data
==========================

.. meta::
   :description: Using MBQC to classify classical data.
   :keywords: mb-qml, mbqc, measurement-based quantum machine learning, qml

**Author(s):** `Luis Mantilla <https://twitter.com/realmantilla>`_

QML can be used to process classical data. Notably, previous studies have applied quantum models 
for such tasks, and you can find more about these in [#havlicek2019]_, [#schuld2019]_, [#abbas2021]_, 
to name a few. In this tutorial, we'll focus on using MB-QML for the task of classical data 
classification. The core idea is to implement an embedding :math:`x_i \mapsto |\phi(x_i)\rangle` 
of a 2D dataset and use this map to formulate the kernel:

.. math:: K(x_i, x_j) = \abs{\braket{\phi(x_i)}{\phi(x_j)}}^2,

The data embedding :math:`\phi` can be implemented in several ways, and in MB-QML, we can use the measurement angles to encode the data. Let's begin creating a simple dataset and splitting it into training and test sets.

.. ipython:: python

   import matplotlib.pyplot as plt
   from sklearn import datasets
   from sklearn.model_selection import train_test_split
   import pandas as pd

   blobs = datasets.make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2)

   blobs_df = pd.DataFrame(data=blobs[0], columns=['feature1', 'feature2'])
   blobs_df['target'] = blobs[1]

   X = blobs_df.drop('target', axis=1)
   y = blobs_df['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   fig, ax = plt.subplots()
   ax.set_facecolor('white') 
   @savefig scatter_classical_data.png width=500px
   plt.scatter(X_train['feature1'], X_train['feature2'], c=y_train, cmap='coolwarm')


It is a good practice to normalize the data to avoid issues when embedding as the measurement angles.

.. ipython:: python

   from sklearn.preprocessing import MinMaxScaler

   scaler = MinMaxScaler(feature_range=(0, 1))
   X_scaler = scaler.fit(X_train)

   X_train = X_scaler.transform(X_train)
   X_test = X_scaler.transform(X_test)

   X_train = np.nan_to_num(X_train)
   X_test = np.nan_to_num(X_test)

Now, we can define the kernel function with an MBQC circuit. We take inspiration from previous 
embeddings mentioned in [#suzuki2020]_ to define the measurement angles for each qubit.

.. code-block:: python

   gs = mp.templates.muta(2,1, one_column=True)
   mp.draw(gs)
   ps = mp.PatternSimulator(gs, backend='numpy-dm', window_size = 4)

   def quantum_kernel(X, Y=None):

      if Y is None:
         Y = X

      K = np.zeros((X.shape[0], Y.shape[0]))
      
      for i, x in enumerate(X):
         angles1 = [x[0], 0,  0,0, x[1], np.cos(x[0])*np.cos(x[1]),0, 0]
         ps.reset()
         state1 = ps(angles1)
         
         for j, y in enumerate(Y):
            angles2 = [y[0], 0,  0,0, y[1], np.cos(y[0])*np.cos(y[1]),0,0]
            ps.reset()
            state2 = ps(angles2)
            
            K[i, j] = mp.calculator.fidelity(state1, state2)
      
      return K


Finally, we can create a SVM classifier and use this kernel to train the model.

.. code-block:: python

   from sklearn import svm
   from sklearn.metrics import accuracy_score

   clf = svm.SVC(kernel=quantum_kernel)
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)

   print("Accuracy:", accuracy_score(y_test, y_pred))


The decision boundary of the trained model can be visualized as follows:

.. code-block:: python

   from matplotlib.colors import ListedColormap

   X_train_np = np.array(X_train)
   y_train_np = np.array(y_train)

   x_min, x_max = X_train_np[:, 0].min() - 0.2, X_train_np[:, 0].max() + 0.2
   y_min, y_max = X_train_np[:, 1].min() - 0.2, X_train_np[:, 1].max() + 0.2
   xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05))

   Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   colors = ('red', 'blue')
   cmap = ListedColormap(colors)

   plt.figure(figsize=(8, 6))
   contour = plt.contourf(xx, yy, 1-Z, alpha=0.4, cmap='coolwarm')
   plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c=1-y_train_np, cmap='coolwarm', edgecolors='k')
   plt.colorbar(contour)

   plt.show()


References
----------

.. [#havlicek2019] Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).

.. [#schuld2019] Schuld, M., & Killoran, N. (2019). Quantum Machine Learning in Feature Hilbert Spaces. Phys. Rev. Lett., 122(4), 040504. 

.. [#abbas2021] Abbas, A., Sutter, D., Zoufal, C. et al. The power of quantum neural networks. Nat Comput Sci 1, 403–409 (2021). 

.. [#suzuki2020] Suzuki, Y., Yano, H., Gao, Q. et al. Analysis and synthesis of feature map for kernel-based quantum classifier. Quantum Mach. Intell. 2, 9 (2020)