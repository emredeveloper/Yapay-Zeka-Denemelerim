from pyvis.network import Network

# Create a simple network
nt = Network('500px', '500px')
nt.add_node(1, label='Node 1')
nt.add_node(2, label='Node 2')
nt.add_edge(1, 2)

# Show the network
nt.show('simple_network.html',notebook=False)