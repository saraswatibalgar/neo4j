from neo4j import GraphDatabase
import csv

uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_graph(tx, row):
    main_col = list(row.keys())[0]
    main_value = row[main_col]

    # Create main node
    tx.run(f"MERGE (m:{main_col} {{name: $main_value}})", main_value=main_value)

    # For other columns, create nodes and relations
    for col, value in row.items():
        if col == main_col:
            continue
        relation = f"REL_{col.upper()}"
        tx.run(f"""
            MERGE (n:{col} {{name: $value}})
            MERGE (m:{main_col} {{name: $main_value}})
            MERGE (m)-[:{relation}]->(n)
        """, main_value=main_value, value=value)

def csv_to_neo4j(file_path):
    with driver.session() as session:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                session.write_transaction(create_graph, row)

csv_file = 'data.csv'
csv_to_neo4j(csv_file)
driver.close()
