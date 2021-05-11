Open this outer folder in visual studie code.
activate the p9deepchem environment. "activate p9deepchem"
Navigate to C:\Users\Andreas\Desktop\P9: "cd C:\Users\Andreas\Desktop\P9"
Run: Python app.Py
Select featurization method and dataset.




Change if's to cases?




There are quite a lot of featurizations. Should I include all of them.



Chose dataset.
Chose Featurizer.
Chose splitter.
Chose model.


Could we use something like ADABOOST in order to use multiple featurizers at the same time?



For mol2vec:
    Do we measure the distance from each of the molecules selected to other molecules and then summarize the distance.
    Or do we make an average from all the selected molecules and then find the distance?
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    https://medium.com/@lope.ai/knn-classifier-from-scratch-with-numpy-python-5c436e26a228

    Takes about 3 secs to find all distances from 1 molecule to all the other molecules in the XML. Would take 8 hours to go through them all.
    So, we should find all the distances beforehand and store them.

    Make a test. Go through the first 20 molecules and save all their distances to other molecules.
    array[MoleculeFrom][MoleculeTo][Distance]

    struct {MoleculeNumber, Average distance} #save everything in a struct that is indexed by the number of the molecule. Make a small for loop that goes
    through the selected molecules, adds the distances together and then substract my the amount of selected molecules and then adds it to the struct as
    the average distance.
    Then sort the array of structs by lowest average distance.




PyFeat:
    python ./Codes/main.py --sequenceType=Protein --testDataset=1 --fasta=./Datasets/Protein/PDB186_independentFASTA.txt --label=./Datasets/Protein/PDB186_independentLabel.txt --kTuple=3 --kGap=5 --pseudoKNC=0 --zCurve=0 --gcContent=0 --cumulativeSkew=0 --atgcRatio=0 --monoMono=0 --monoDi=0 --monoTri=0 --diMono=0 --diDi=0 --diTri=0 --triMono=1 --triDi=0



Currently looking up negative SMILES
Featurize negative molecules - get problematic indicies for negative molecules

Look up Negative Molecules -> Size X                                                                                    IN PROGRESS
Featurize the Molecules -> Size X with empty Strings                                                                    TODO
Look up the Negative Proteins -> Size Y because some can't be looked up -> List of problematic PROTEIN indices          DONE
Remove Problematic PROTEIN indices from FEATURIZED MOLECULES -> Size Y with empty Strings
Recalculate problematic Molecular indices from empty Strings
Remove from Both FEATURIZED MOLECULES and LOOKED UP PROTEINS -> Size Z for both without empty strings
Featurize Z proteins

Important. Remove from end to beginning.

TODO: Revise app.py, experiments.py so that they don't use aau40000 dataset
