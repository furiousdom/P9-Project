

def load_dataset(dataset_name):
    molecules = pd.read_csv(f'./data/datasets/{dataset_name}/molecules.csv')
    proteins = pd.read_csv(f'./data/datasets/{dataset_name}/proteins.csv')
    return molecules, proteins

def concat_mol_prot(molecules_data_frame, protein_data_frame):
    return pd.concat([molecules_data_frame, protein_data_frame], axis=1)

def create_aau_output(df):
    Y = np.zeros((df.shape[0], 2), dtype=int)
    Y[:1000] = [0, 1]
    Y[1000:] = [1, 0]
    return Y

def load_train_dataset():
    aau_molecules, aau_proteins = load_dataset('aau')
    aau_X = pd.concat([aau_molecules, aau_proteins], axis=1)
    aau_X = np.array(aau_X)
    aau_Y = create_aau_output(aau_X)
    return aau_X, aau_Y

def load_test_datasets():
    kiba_molecules, kiba_proteins = load_dataset('kiba')
    davis_molecules, davis_proteins = load_dataset('davis')

    kiba = pd.concat([kiba_molecules, kiba_proteins], axis=1)
    davis = pd.concat([davis_molecules, davis_proteins], axis=1)

    kiba_X = np.array(kiba)
    davis_X = np.array(davis)

    kiba_Y = np.array(load_binary_scores('./data/datasets/kiba/scores.txt', 12.1))
    davis_Y = np.array(load_binary_scores('./data/datasets/davis/scores.txt', 7.0, True))
    return kiba_X, kiba_Y, davis_X, davis_Y