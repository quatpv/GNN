import numpy as np


class BuildGraph():
    def __init__(self, num_nodes, start_index):
        """
        Args:
            num_nodes (int): Number node of the graph
            data (tensor): NxD - N is number of nodes, D is dimension of node features
        """
        self.num_nodes = num_nodes
        self.start_index = start_index

    def font_simple_graph(self):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for u in range(self.num_nodes - 1):
            source = np.append(source, u)
            destination = np.append(destination, u + 1)

        source += self.start_index
        destination += self.start_index

        return [source, destination]

    def back_simple_graph(self):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for u in range(1, self.num_nodes):
            source = np.append(source, u)
            destination = np.append(destination, u - 1)

        source += self.start_index
        destination += self.start_index
        
        return [source, destination]

    def font_dense_graph(self):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for u in range(self.num_nodes - 1):
            for v in range(u + 1, self.num_nodes):
                source = np.append(source, u)
                destination = np.append(destination, v)

        source += self.start_index
        destination += self.start_index

        return [source, destination]

    def back_dense_graph(self):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for u in range(1, self.num_nodes):
            for v in range(u - 1):
                source = np.append(source, u)
                destination = np.append(destination, v)
        
        source += self.start_index
        destination += self.start_index

        return [source, destination]

    def complete_graph(self):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v:
                    source = np.append(source, u)
                    destination = np.append(destination, v)
        
        source += self.start_index
        destination += self.start_index

        return [source, destination]