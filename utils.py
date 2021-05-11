import json
import psycopg2
import pandas as pd

def load_json_obj_from_file(file_name):
    json_file = open(file_name, 'r', encoding='utf-8')
    json_obj = json.load(json_file)
    json_file.close()
    return json_obj

def save_json_obj_to_file(file_name, obj):
    json_file = open(file_name, 'w', encoding='utf-8')
    data = json.dumps(obj)
    json_file.write(data)
    json_file.close()

def save_items_to_txt_by_line(file_name, items):
    with open(file_name, 'w', encoding="utf-8") as file:
        for item in items:
            file.write(f'{item}\n')

def save_labels_to_txt_file(file_name, num_of_samples, num_of_positive_samples):
    with open(file_name, 'w', encoding='utf-8') as label_file:
        for i in range(num_of_samples):
            if i < num_of_positive_samples:
                label_file.write('1\n')
            else:
                label_file.write('0\n')

def json_dataset_to_csv(dataset_name):
    json_dataset = load_json_obj_from_file('./data/' + dataset_name + '.json')
    columns = ['FASTA', 'SMILES', 'BINDING AFFINITY']
    if dataset_name == 'davis':
        columns = ['SMILES', 'FASTA', 'BINDING AFFINITY']
    df = pd.DataFrame(json_dataset, columns=columns)
    df.to_csv('./data/' + dataset_name + '.csv')

def extract_smiles_list_from_pairs(pairs):
    return [pair[0] for pair in pairs]

def extract_fasta_list_from_pairs(pairs):
    return [pair[1] for pair in pairs]

def save_fasta_for_pyfeat(file_name, smiles_fasta_pairs):
    fasta_list = extract_fasta_list_from_pairs(smiles_fasta_pairs)
    save_items_to_txt_by_line(file_name, fasta_list)

def save_smiles_for_featurizer(file_name, smiles_fasta_pairs):
    smiles_list = extract_smiles_list_from_pairs(smiles_fasta_pairs)
    save_json_obj_to_file(file_name, smiles_list)

def save_molecule_embeddings_to_csv(file_name, molecule_embeddings):
    molecules_data_frame = pd.DataFrame(molecule_embeddings)
    molecules_data_frame.to_csv(file_name)

def concatenate_dataset(dataset_name):
    dataset_path = f'./data/datasets/{dataset_name}/'
    molecules_all = pd.read_csv(dataset_path + 'molecules_all.csv')
    molecules_rest = pd.read_csv(dataset_path + 'molecules_rest.csv')
    molecules = pd.concat([molecules_all, molecules_rest], axis=0)
    molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)

    proteins_all = pd.read_csv(dataset_path + 'proteins_all.csv')
    proteins_rest = pd.read_csv(dataset_path + 'proteins_rest.csv')
    proteins = pd.concat([proteins_all, proteins_rest], axis=0)
    proteins.drop(proteins.filter(regex="Unname"),axis=1, inplace=True)

    molecules.to_csv(dataset_path + 'molecules.csv')
    proteins.to_csv(dataset_path + 'proteins.csv')

def connect_db():
    db_session = psycopg2.connect(
        host='localhost',
        port='5432',
        dbname='drugdb',
        user='postgres',
        password='postgres'
    )
    return db_session.cursor()

def print_df_shape_head_tail(df, count):
    print('Shape:\n')
    print(df.shape)
    print()
    print('Head:\n')
    print(df.head(count))
    print()
    print('Tail:\n')
    print(df.tail(count))

def read_fastas_from_file(file_name):
    '''
    :param file_name:
    :return: genome sequences (FASTA strings)
    '''
    with open(file_name, 'r') as file:
        sequences = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                sequences.append(genome.upper())
                genome = ''
        sequences.append(genome.upper())
        del sequences[0]
        return sequences

def make_aau_scores_file(dataset_name, total_number, negative_count):
    file_path = f'./data/datasets/{dataset_name}/binding_affinities.txt'
    with open(file_path, 'w') as file:
        for i in range(total_number):
            if i < total_number - negative_count: file.write('1\n')
            else: file.write('0\n')

def save_predictions(dataset_name, y_test, predictions):
    with open(f'./data/results/{dataset_name}-results.txt', 'w') as file:
        for i in range(len(y_test)):
            file.write(f'{y_test[i]} {predictions[i]}\n')

