import math

import networkx as nx
import matplotlib.pyplot as plt
# import plotly.graph_objects as go


def visualize_connections(characters, connections):
    """
    :param characters: dictionary of characters and their sentiments
    :param connections: dictionary of connections and their weights
    """

    # Visualize sentiment scores in graph
    if characters is not None and characters != {}:
        # Create graph
        G = nx.Graph()

        # Create nodes
        for c, sentiment in characters.items():
            G.add_node(c, sentiment=sentiment)

        # Create edges
        for c1, c2 in connections:
            weight = connections[c1, c2]
            G.add_edge(c1, c2, weight=weight)

        node_colors = [G.nodes[node]['sentiment'] for node in G.nodes]
        edge_widths = [math.log2(G.edges[edge]['weight']) + 1 for edge in G.edges]

        # Create colormap
        cmap = plt.cm.get_cmap('RdYlGn')
        edge_cmap = plt.cm.Blues

        # Set positions of nodes
        pos = nx.circular_layout(G)

        # Draw nodes with colors based on sentiments
        nx.draw(G, pos,
                node_color=node_colors, edge_color=edge_widths,
                cmap=cmap, edge_cmap=edge_cmap,
                width=2.0, with_labels=True, node_size=200)

        # Draw edges with widths based on weights of connections
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray')

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_color='black')

        # Show the plot
        plt.axis('off')
        plt.show()

    else:
        print('No (main) characters found in the book.')

"""  Legacy code for interactive graph visualization using Plotly
edge_x = []
edge_y = []
x_text = []
y_text = []
weights = []
# Add edges to edge trace
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    x_text.append((x0 + x1) / 2)
    y_text.append((y0 + y1) / 2)
    weights.append(G.edges[edge]['weight'])
    # color.append(G.edges[edge]['weight']) TODO - display connection weights

# Create edge traces
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create text trace
text_trace = go.Scatter(
    x=x_text, y=y_text,
    mode='text',
    text=weights,
    textposition="top center")

node_x = []
node_y = []
# Add node info to node trace
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

# Create node trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=characters.keys(),
    marker=dict(
        showscale=True,
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],  # TODO - display sentiment scores
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Create figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Character Connections',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

# Show graph
fig.show()
"""
