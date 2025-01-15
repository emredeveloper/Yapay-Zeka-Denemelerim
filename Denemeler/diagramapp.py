from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
from sqlalchemy import create_engine
from sqlalchemy.inspection import inspect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

engine = create_engine('sqlite:///mydatabase.db')  # Replace with your database URL
inspector = inspect(engine)

tables = []
for table_name in inspector.get_table_names():
    columns = inspector.get_columns(table_name)
    processed_columns = []
    for col in columns:
        processed_columns.append({
            'name': col['name'],
            'type': str(col['type']),
            'nullable': col['nullable']
        })
    foreign_keys = inspector.get_foreign_keys(table_name)
    fks = [fk['referred_table'] for fk in foreign_keys]
    tables.append({
        'name': table_name,
        'columns': processed_columns,
        'foreign_keys': fks
    })

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Database Diagram</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
        <style>
            #cy {
                width: 100%;
                height: 100vh;
                position: absolute;
                top: 0;
                left: 0;
            }
        </style>
    </head>
    <body>
        <div id="cy"></div>
        <script>
            fetch('/schema')
                .then(response => response.json())
                .then(data => {
                    var elements = [];
                    data.forEach(table => {
                        elements.push({
                            group: 'nodes',
                            data: {
                                id: table.name,
                                label: table.name,
                                columns: table.columns.map(col => col.name)
                            }
                        });
                    });
                    data.forEach(table => {
                        table.foreign_keys.forEach(targetTable => {
                            elements.push({
                                group: 'edges',
                                data: {
                                    source: table.name,
                                    target: targetTable
                                }
                            });
                        });
                    });
                    var cy = cytoscape({
                        container: document.getElementById('cy'),
                        elements: elements,
                        style: [
                            {
                                selector: 'node',
                                style: {
                                    'shape': 'rectangle',
                                    'width': 'label',
                                    'height': 'label',
                                    'padding': 12,
                                    'text-valign': 'center',
                                    'background-color': '#69b3a2',
                                    'color': 'white',
                                    'font-size': 14
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'curve-style': 'bezier',
                                    'width': 2,
                                    'line-color': '#ccc',
                                    'target-arrow-color': '#ccc',
                                    'target-arrow-shape': 'triangle'
                                }
                            }
                        ],
                        layout: {
                            name: 'dagre'
                        }
                    });
                    cy.on('tap', 'node', function(event){
                        var node = event.target;
                        var name = node.data('id');
                        var columns = node.data('columns');
                        alert('Table: ' + name + '\\nColumns: ' + columns.join(', '));
                    });
                });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/schema')
def schema():
    return jsonify(tables)

if __name__ == '__main__':
    socketio.run(app, debug=True)