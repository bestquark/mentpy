An introduction to MBQC
========================

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction

Measurement-based quantum computation is a paradigm of quantum computation that uses 
single qubit measurements to perform universal quantum computation. It is equivalent to 
the standard gate-based model. The main difference is that the gates are not explicitly
applied to the qubits, but rather a big resource (entangled) state is prepared and logical 
information flows through the system by measuring the qubits of the resource state. 

In :obj:`mentpy` we can simulate an MBQC circuit by using the :obj:`MBQCCircuit` class.

.. ipython:: python

    gs = mp.GraphState()
    gs.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    mbcirc = mp.MBQCState(gs, input_nodes = [0], output_nodes =[4])
    print(mbcirc)

We can concatenate two MBQC circuits with the :func:`merge`, :func:`hstack`, or :func:`vstack`
functions. 

.. md-tab-set::
    .. md-tab-item:: merge

        .. ipython:: python

            new_circ = mp.merge(mbcirc, mbcirc, along=[(4,0)])  # specify nodes to merge
            plt.figure(figsize=(8,2));
            @savefig merge_mbqc.png width=4in
            mp.draw(new_circ, with_labels = True, fix_wires = [tuple(range(10))])
    
    .. md-tab-item:: hstack
            
        .. ipython:: python

            new_circ = mp.hstack((mbcirc, mbcirc))
            plt.figure(figsize=(8,2));
            @savefig hstack_mbqc.png width=4in
            mp.draw(new_circ, with_labels = True, fix_wires = [tuple(range(10))])
        
    .. md-tab-item:: vstack

        .. ipython:: python

            new_circ = mp.vstack((mbcirc, mbcirc))
            plt.figure(figsize=(8,3));
            @savefig vstack_mbqc.png width=4in
            mp.draw(new_circ, with_labels = True, fix_wires = [(0,1,2,3,4), (5,6,7,8,9)])

