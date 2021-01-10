# Graph class
import torch
import numpy as np
try:
    import torch_geometric
except:
    torch_geometric = None


class Graph(object):
    def __init__(self, molecules, all_atomic_type, construct_adj=True):
        self.molecules = molecules

        self.nMolecules = len(self.molecules)
        self.nVertices = 0
        self.nEdges = 0
        for mol in range(self.nMolecules):
            self.nVertices += self.molecules[mol].nAtoms
            for atom in range(self.molecules[mol].nAtoms):
                self.nEdges += self.molecules[mol].atoms[atom].nNeighbors

        # Vertex indexing
        self.start_index = np.zeros((self.nMolecules + 1), dtype=np.int32)
        count = 0
        for mol in range(self.nMolecules):
            self.start_index[mol] = count
            count += self.molecules[mol].nAtoms
        assert count == self.nVertices
        self.start_index[self.nMolecules] = self.nVertices
        self.start_index = torch.from_numpy(self.start_index)

        # Adjacency matrix (total of all smaller molecules)
        if construct_adj:
            self.adj = np.zeros((self.nVertices, self.nVertices), dtype=np.float32)

        # Edge indexing
        self.edges_tensor = np.zeros((self.nEdges, 2), dtype=np.float32)
        count = 0
        for mol in range(self.nMolecules):
            for atom in range(self.molecules[mol].nAtoms):
                u = self.start_index[mol] + atom
                for i in range(self.molecules[mol].atoms[atom].nNeighbors):
                    v = self.start_index[mol] + self.molecules[mol].atoms[atom].neighbors[i]
                    self.edges_tensor[count, 0] = u
                    self.edges_tensor[count, 1] = v
                    count += 1

                    # Adjacency matrix
                    if construct_adj:
                        self.adj[u, v] = 1.0
                        self.adj[v, u] = 1.0
        assert count == self.nEdges
        self.edges_tensor = torch.from_numpy(self.edges_tensor)
        if construct_adj:
            self.adj = torch.from_numpy(self.adj)

        # Feature indexing
        nAtomicTypes = len(all_atomic_type)
        x = []
        y = []
        v = []
        for mol in range(self.nMolecules):
            for i in range(self.molecules[mol].nAtoms):
                index = all_atomic_type.index(self.molecules[mol].atomic_type[i])
                x.append(self.start_index[mol] + i)
                y.append(index)
                v.append(1.0)
        print("x: ", torch.stack(x))
        print("y: ", y)
        print("v: ", v)
        print("edges_tensor: ", self.edges_tensor)
        index_tensor = torch.LongTensor([x, y])
        value_tensor = torch.FloatTensor(v)
        self.feature = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nVertices, nAtomicTypes]))
        print('self.feature', self.feature)

        # Label indexing
        self.label = []
        for mol in range(self.nMolecules):
            self.label.append(self.molecules[mol].class_)
        self.label = np.array(self.label)
        print('self.label', self.label)
        raise Exception


class TGMGraph(object):
    def __init__(self, batch, construct_adj=True, nAtomicTypes=None):
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
        if nAtomicTypes is None:
            nAtomicTypes = torch.max(batch.x)+1
        self.nAtomicTypes = nAtomicTypes

        # Batch properties
        self.start_index = []
        last_val = 0
        for i, val in enumerate(batch.batch):
            if val > last_val:
                self.start_vals.append(i)
                last_val == val
        self.start_index = torch.tensor(self.start_index).long()

        # Vertex Properties
        index_tensor = torch.LongTensor([torch.arange(self.nVertices),
                                         batch.x.squeeze(1)])
        self.feature = torch.sparse.FloatTensor(index_tensor, torch.ones(self.nVertices), torch.Size([self.nVertices, nAtomicTypes]))

        # Edge properties
        self.edges_tensor = batch.edge_index.T
        self.label = batch.y

        # Construct optional adjacency matrix
        if torch_geometric is None:
            raise ImportError("torch_geometric not found")
        if construct_adj:
            self.adj = torch_geometric.utils.to_dense_adj(batch.edge_index)
