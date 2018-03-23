from Bio import SeqIO


def hydrophobicity_dictionary():
    '''
    Creates a dictionary mapping each amino acid
    to its hydrophobicity.
    '''
    hydro_list = [('I', 4.5), ('V', 4.2), ('L', 3.8), ('F', 2.8), ('C', 2.5), ('M', 1.9), ('A', 1.8), ('G', -0.4), ('T', -0.7), ('S', -0.8), ('W', -0.9),
                  ('Y', -1.3), ('P', -1.6), ('H', -3.2), ('E', -3.5), ('Q', -3.5), ('D', -3.5), ('N', -3.5), ('K', -3.9), ('R', -4.5), ('U', 0), ('X', 0), (' ', 0), ('B', 0)]

    hydro_dict = dict()
    for element in hydro_list:
        amino_acid, hydro_value = element
        hydro_dict[amino_acid] = hydro_value

    return hydro_dict

hydro_dict = hydrophobicity_dictionary()

def hydrophobicity_encode(seq):
    '''
    Given a sequence, this method calculates the
    total hydrophobicity of the protein.
    '''
    hydro_phobicity = 0

    for letter in seq:
        hydro_phobicity += hydro_dict[letter]

    return hydro_phobicity


def read_data():
    '''
    Returns the sequences and labels for all protein localisation
    classes.
    '''
    cyto_dict = SeqIO.to_dict(SeqIO.parse('./data/cyto.fasta', 'fasta'))
    mito_dict = SeqIO.to_dict(SeqIO.parse('./data/mito.fasta', 'fasta'))
    nucleus_dict = SeqIO.to_dict(SeqIO.parse('./data/nucleus.fasta', 'fasta'))
    secreted_dict = SeqIO.to_dict(
        SeqIO.parse('./data/secreted.fasta', 'fasta'))

    sequences = []
    labels = []

    for key, value in cyto_dict.items():
        sequences.append(value.seq.tostring())
        labels.append(0)

    for key, value in mito_dict.items():
        sequences.append(value.seq.tostring())
        labels.append(1)

    for key, value in nucleus_dict.items():
        sequences.append(value.seq.tostring())
        labels.append(2)

    for key, value in secreted_dict.items():
        sequences.append(value.seq.tostring())
        labels.append(3)

    return sequences, labels
