import pickle
from Bio import SeqIO
from classical_main import sequence_features
from sklearn.feature_extraction import DictVectorizer

'''
Reads in blind test set, performing a classification on the data using a saved model.
'''

blind_test_dict = SeqIO.to_dict(SeqIO.parse('./data/blind_test', 'fasta'))

vectorizer = DictVectorizer()
identifiers = [value.id for value in blind_test_dict.values()]
class_map = {0: 'Cyto', 1: 'Mito', 2: 'Nuclear', 3: 'Secreted'}

sequences = [sequence for sequence in [value.seq.tostring() for value in blind_test_dict.values()]]
encoded_sequences = vectorizer.fit_transform([sequence_features(sequence) for sequence in sequences])
model = pickle.load(open('model', 'rb'))

predictions = model.predict(encoded_sequences)
confidences = np.max(model.predict_proba(encoded_sequences), axis=1)

for i in range(len(sequences)):
    print(identifiers[i], class_map[predictions[i]], confidences[i])
