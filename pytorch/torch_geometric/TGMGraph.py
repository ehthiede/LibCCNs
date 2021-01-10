import torch
try:
    import torch_geometric
except ImportError:
    torch_geometric = None


class TGMGraph(object):
    def __init__(self, batch, construct_adj=True):
        """
        Initializes a the graph from a TGM data object.

        Arguments
        ---------
        batch: Batch object
            The torch geometric batch object we are grabbing the data from.
        construct_adj : bool
            If True, construct the (dense) adjacency matrix.  This is
            False by default, since the adjacency matrix can be expensive.
        """
        self.nMolecules = batch.num_graphs
        self.nVertices = batch.num_nodes
        self.nEdges = batch.num_edges

        # Batch properties
        self.start_index = []
        last_val = 0
        for i, val in enumerate(batch.batch):
            if val > last_val:
                self.start_vals.append(i)
                last_val == val
        self.start_index = torch.tensor(self.start_index).long()

        # Vertex Properties
        self.x = batch.x

        # Edge properties
        self.edges_tensor = batch.edge_index.T
        self.label = batch.y

        # Construct optional adjacency matrix
        if torch_geometric is None:
            raise ImportError("torch_geometric not found")
        if construct_adj:
            self.adj = torch_geometric.utils.to_dense_adj(batch.edge_index)
