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
    @savefig 1d_cluster.png width=1000px
    mp.draw(mbcirc)

This circuit is designed to implement a single qubit gate on the input qubit, with qubits 
measured in a left-to-right sequence. We can use this same circuit to implement a teleportation by 
measuring qubits :math:`0`, :math:`1`, :math:`2`, and :math:`3` in the :math:`X` basis. However,
the output qubit with a byproduct operator :math:`Z` that depends on the earlier qubits measurement 
outcomes.

To build this circuit, we can use the :obj:`Measurement` (or its alias :obj:`Ment`) operator to the 
circuit's qubits. The :obj:`Measurement` object is characterized by an angle and a measurement 
plane, which can be "XY", "XZ", or "YZ" ("X", "Y", and "Z" are also accepted), determining the 
basis in which the qubit is measured. The angle can be specified or set to ``None``. 
If not provided, it will be treated as a trainable parameter.

.. ipython:: python

    mbcirc[0] = mp.Ment('X')
    mbcirc[1] = mp.Ment('X')
    mbcirc[2] = mp.Ment('X')
    mbcirc[3] = mp.Ment('X')
    @savefig 1d_cluster_measure.png width=1000px
    mp.draw(mbcirc, label='planes')

Similarly, we can define the same circuit when creating the :obj:`MBQCircuit` object.

.. ipython:: python

    mbcirc = mp.MBQCircuit(gs, input_nodes=[0], output_nodes=[4], 
                           measurements={i: mp.Ment('X') for i in range(4)})
    @savefig 1d_cluster_measure2.png width=1000px
    mp.draw(mbcirc, label='arrows')

If not every measurement is specified, the unspecified measurements will be set to the
value of the :obj:`default_measurement` attribute of the :obj:`MBQCircuit` object. By default,
it is set to :math:`XY`, but we can change it to any other :obj:`Ment` object. Let's see an example:

.. ipython:: python

    mbcirc = mp.MBQCircuit(gs, input_nodes=[0], output_nodes=[4], 
                           measurements={1: mp.Ment('XY')},
                           default_measurement=mp.Ment('X'))
    print(mbcirc[0]) # Not specified in the constructor
    print(mbcirc[1]) # Specified in the constructor
    print(mbcirc[2].matrix()) # Matrix of the measurement operator

We can concatenate two MBQC circuits with the :func:`merge`, :func:`hstack`, or :func:`vstack`
functions. 

.. md-tab-set::
    .. md-tab-item:: merge

        .. ipython:: python

            new_circ = mp.merge(mbcirc, mbcirc, along=[(4,0)])  # specify nodes to merge
            @savefig merge_mbqc.png width=1000px
            mp.draw(new_circ, label='angles')
    
    .. md-tab-item:: hstack
            
        .. ipython:: python

            new_circ = mp.hstack((mbcirc, mbcirc))
            @savefig hstack_mbqc.png width=1000px
            mp.draw(new_circ, label='planes')
        
    .. md-tab-item:: vstack

        .. ipython:: python

            new_circ = mp.vstack((mbcirc, mbcirc))
            @savefig vstack_mbqc.png width=1000px
            mp.draw(new_circ, label='arrows')

To use pre-defined MBQC circuits, we can use the :obj:`templates` module, which contains
some common MBQC circuits. For example, we can create a grid cluster state with the 
:func:`grid_cluster` function.

.. ipython:: python

    grid_cluster = mp.templates.grid_cluster(3, 5)
    linear_cluster = mp.templates.linear_cluster(4)
    grid_and_linear = mp.merge(grid_cluster, linear_cluster, along=[(9,0)])
    @savefig template_merge.png width=1000px
    mp.draw(grid_and_linear)

Finally, if you want to know the set of gates that the MBQC circuit you have created
implements, you can use the :func:`utils.calculate_lie_algebra`. This function returns
the lie algebra :math:`\fraktur{g}` that the circuit implements, which can be used to calculate the set of 
gates using the exponential map :math:`e^{\fraktur{g}}`.

.. ipython:: python
    :okwarning:

    ops = mp.utils.calculate_lie_algebra(grid_cluster)
    print(len(ops))
    ops[:3]