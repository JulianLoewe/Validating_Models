import numpy as np
from dtreeviz.trees import DTreeViz
import tempfile
import os
from validating_models.stats import get_decorator

time_picture_coposition = get_decorator('picture_composition')

tmp = tempfile.gettempdir()

class DTreeVizConv(DTreeViz):
    def __init__(self, dot, scale=1.0):
        super().__init__(dot, scale)
    
    @staticmethod
    def from_DTreeViz(dtreeviz):
        return DTreeVizConv(dtreeviz.dot, dtreeviz.scale)

    @time_picture_coposition
    def save(self, filename, transparent = True, dpi=1000):

        dot_idx = filename.rindex('.')
        svg_file = filename[:dot_idx] + '.svg'
        super().save(svg_file)
        
        # Add missing header
        missing_header = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        with open(svg_file, 'r') as original:
            svg_content = original.read()
        
        if filename.endswith('.svg'):
            with open(filename, 'w') as new:
                new.write(missing_header)
                new.write(svg_content)
        else:
            import cairosvg
            import io
            from PIL import Image
            output_png = io.BytesIO()
            cairosvg.svg2png(bytestring=svg_content, write_to=output_png, scale=dpi/100)
            im = Image.open(output_png)
            output_png.close()

            if transparent:
                im = im.convert('RGBA')
                data = np.array(im)
                #print(data.shape)
                rgb = data[:,:,:3]
                white = [255,255,255]
                transparent = [255,255,255,0]
                mask = np.all(rgb == white, axis = -1)
                #print(np.sum(mask))
                data[mask] = transparent
                im = Image.fromarray(data)

            im.save(filename)



def html_label(label, color, size):
    '''Returns the html representation to show a label in a html table.
    '''
    return f'<font face="Helvetica" color="{color}" point-size="{size}"><i>{label}</i></font>'

def html_image(html_label, img_path):
    '''Returns the html representation to show the label above the given image.
    '''
    html_label_row = f'<tr><td CELLPADDING="0" CELLSPACING="0">{html_label}</td></tr>'
    return f"""<table border="0" CELLBORDER="0">
        {html_label_row}
        <tr>
                <td><img src="{img_path}"/></td>
        </tr>
        </table>"""

def html_node_label(node, color, size):
    return html_label(f"Node {node.id}",color, size)

def node_stmt(node_name, html_content, highlight:bool, colors):
    if highlight:
        return f'{node_name} [margin="0" shape=box penwidth=".5" color="{colors["highlight"]}" style="dashed" label=<{html_content}>]' 
    else: 
        return f'{node_name} [margin="0" shape=box penwidth="0" color="{colors["text"]}" label=<{html_content}>]'

def cluster_nodes(cluster_name, label, nodes):
    newline = "\n\t"
    return f'''subgraph cluster_{cluster_name} {{
        color=lightgrey;
        label="{label}";
        {newline.join(nodes)}
    }}
    '''

def class_legend_gr(path):
    if path != None:
        return f'''subgraph cluster_legend {{
                style=invis;
                legend [penwidth="0" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {html_image('', path)}
                >]
            }}
            '''
    else:
        return ''

def get_image_path(identifier: str):
    return os.path.join(tmp,f"{tmp}/{identifier}_{os.getpid()}.svg")

def grid_layout(title, nodes, edges, legend, colors, size=None, fontname='Helvetica', scale=1.0, orientation = 'LR'): # LR or TD
    node_names = [node.split(" ",1)[0] if node.split(" ",1)[0] != 'subgraph' else node.split(" ",2)[1] for node in nodes]
    if size == None:
        grid_size = int(np.ceil(np.sqrt(len(node_names))))
        size = (grid_size,grid_size)

    size = list(size)
    num_nodes = size[0] * size[1]

    big_index = 0 if size[0] > size[1] else 1

    while num_nodes >= len(nodes) + size[big_index]:
        size[big_index] = size[big_index] - 1
        num_nodes = size[0] * size[1]
    
    while num_nodes >= len(nodes) + size[(big_index + 1) % 2]:
        size[(big_index + 1) % 2] = size[(big_index + 1) % 2] - 1
        num_nodes = size[0] * size[1]

    # Fill grid with spaceholder nodes
    num_spaceholders = 0
    while len(nodes) < num_nodes:
        num_spaceholders += 1
        nodes.append(node_stmt(f'spaceholder{num_spaceholders}', '', False, colors))
        node_names.append(f'spaceholder{num_spaceholders}')

    node_names = np.array(node_names).reshape((size[0],size[1]))
    # Create grid structure
    grid_edges = []
    for i in range(size[0]): 
        # Horizontal Connections
        vec = node_names[i,:]
        vec = list(vec.reshape(-1,))
        if len(vec) > 1:
            grid_edges.append('rank=same {' + ' -- '.join(vec) + '}')
    
    for j in range(size[1]):
        # Vertical Connections
        vec = node_names[:,j]
        vec = list(vec.reshape(-1,))
        if len(vec) > 1:
            grid_edges.append(' -- '.join(vec))
        
        if orientation == 'LR' and j == 0 and legend != None:
            grid_edges.append(' -- legend')


    newline = "\n\t"
    dot = f'''
    graph G {{
        splines=line;
        fontname="{fontname}"
        node [fontname="{fontname}"]
        edge [fontname="{fontname}"]
        layout=dot
        label="{title}"
        labelloc="t"
        rankdir="{orientation}"

        {newline.join(nodes)}

        {newline.join(edges)}

        edge [style=invis]

        {newline.join(grid_edges)}
        {class_legend_gr(legend) if legend != None else ''}

    }}
    '''
    return DTreeVizConv(dot, scale)