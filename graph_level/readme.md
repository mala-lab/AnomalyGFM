there are two problems currently:

    1. for graph-level AD, the graphs are fed into the GNNs via batch manner. However, due to the scarcity of abnormal data, there exists a scenario where all the graphs in a batch are normal.
    2. how to construct the graph of graph

the code has some bugs to be fixed.
