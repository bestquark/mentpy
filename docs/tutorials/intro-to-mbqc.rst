An introduction to MBQC
========================


Measurement-based quantum computation is a paradigm of quantum computation that uses 
single qubit measurements to perform universal quantum computation. It is equivalent to 
the standard gate-based model. The main difference is that the gates are not explicitly
applied to the qubits, but rather a big resource (entangled) state is prepared and logical 
information flows through the system by measuring the qubits of the resource state. 

In :obj:`mentpy` we can simulate an MBQC circuit by using the :obj:`MBQCircuit` class.

.. ipython:: python

    gs = mp.GraphState()
    gs.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    mbcirc = mp.MBQCircuit(gs, input_nodes=[0], output_nodes=[4])
    print(mbcirc)

We can concatenate two MBQC circuits with the :func:`merge`, :func:`hstack`, or :func:`vstack`
functions. 

.. md-tab-set::
    .. md-tab-item:: merge

        .. ipython:: python

            new_circ = mp.merge(mbcirc, mbcirc, along=[(4,0)])  # specify nodes to merge
            @savefig merge_mbqc.png width=1000px
            mp.draw(new_circ)
    
    .. md-tab-item:: hstack
            
        .. ipython:: python

            new_circ = mp.hstack((mbcirc, mbcirc))
            @savefig hstack_mbqc.png width=1000px
            mp.draw(new_circ)
        
    .. md-tab-item:: vstack

        .. ipython:: python

            new_circ = mp.vstack((mbcirc, mbcirc))
            @savefig vstack_mbqc.png width=1000px
            mp.draw(new_circ)

To use pre-defined MBQC circuits, we can use the :obj:`templates` module. 

.. ipython:: python

    grid_cluster = mp.templates.grid_cluster(3, 5)
    linear_cluster = mp.templates.linear_cluster(4)
    grid_and_linear = mp.merge(grid_cluster, linear_cluster, along=[(9,0)])
    @savefig template_merge.png width=1000px
    mp.draw(grid_and_linear)
    