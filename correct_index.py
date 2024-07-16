from pathlib import Path
import sys

from Bio import Seq, SeqIO, SeqRecord
from Bio.PDB import PDBList, PDBParser, MMCIF2Dict
import numpy as np
import pandas as pd

from PDAnalysis import pdb_parser
from PDAnalysis.protein import Protein


PATH_PDB = Path("/home/johnmcbride/projects/ProteinEvolution/AFsequences/ddG/PDB")
PATH_FASTA = Path("/home/johnmcbride/projects/ProteinEvolution/AFsequences/ddG/fasta")


def download_proteins():
    df = pd.read_csv('../Data/thermomutdb.csv')
    pdbl = PDBList()
#   for pdb in df.loc[df.mutation_based=='PDB', 'PDB_wild'].unique():
    for pdb in df.loc[df.PDB_wild.notnull(), 'PDB_wild'].unique():
        if not PATH_PDB.joinpath(f"{pdb.lower()}.cif").exists():
            pdbl.retrieve_pdb_file(pdb, pdir=PATH_PDB)


def compare_indices(uniprot, mutation, pdb, chain):
    print(uniprot, mutation, pdb, chain)
    path_uniprot = PATH_FASTA.joinpath(f'../fasta/{uniprot}.fasta')
    path_pdb = PATH_PDB.joinpath(f"{pdb.lower()}.cif")
    mut_i = int(mutation[1:-1]) - 1

    seq_uniprot = str(list(SeqIO.parse(path_uniprot, 'fasta'))[0].seq)

    try:
        prot = Protein(path_pdb, chain = chain, pdb_fill_missing_nan=True)
    except:
        return False, -1

    try:
        if mutation[0] == prot.sequence[mut_i]:
            return True, match_uniprot_index(seq_uniprot, prot.sequence, mut_i)
    except Exception as e:
        print(e)


    return False, -1


def match_uniprot_index(seq_uniprot, seqres, mut_i):
    candidates = pdb_parser.align_sequences(seq_uniprot, ''.join(seqres))
    if len(candidates) == 0:
        return -1

    elif len(candidates) == 1:
        al = np.array(list(candidates[0]))
        not_gap = al != '-'
        return np.where(np.cumsum(not_gap.astype(int)) == mut_i + 1)[0][0]

    else:
        print(f"{len(candidates)} candidate alignments found!")
        sites = set()
        for c in candidates:
            al = np.array(list(c))
            not_gap = al != '-'
            sites.add(np.where(np.cumsum(not_gap.astype(int)) == mut_i + 1)[0][0])
        # If there is no ambiguity, then return the index
        if len(sites) == 1:
            return list(sites)[0]
        else:
            return -1


def convert_mutation_code_to_uniprot(df):
    cols = ['uniprot', 'mutation_code', 'PDB_wild', 'mutated_chain']
    vals = df.loc[df.mutation_based=='PDB', cols].values

    # Find uniprot indices that match the PDB indices
    is_found, idx = np.array([compare_indices(*v) for v in vals]).T

    # Exclude any cases linked to PDB files with problems
    pdb_exclude = np.unique(vals[~is_found.astype(bool)][:,2])
    to_use = np.in1d(vals[:,2], pdb_exclude) == False

    df.loc[df.mutation_based!='PDB', 'mut_idx_uniprot'] = df.loc[df.mutation_based!='PDB', 'mutation_code'].apply(lambda x: int(x[1:-1]))
    df.loc[df.mutation_based!='PDB', 'to_use'] = True

    df.loc[df.mutation_based=='PDB', 'mut_idx_uniprot'] = idx + 1
    df.loc[df.mutation_based=='PDB', 'to_use'] = to_use
    
    return df.loc[df.to_use]


def load_mmcif(path):
    return MMCIF2Dict.MMCIF2Dict(path)


def get_pdb_oligomer(pdb):
    mmcif = load_mmcif(pdb)
    try:
        return mmcif['_pdbx_struct_assembly.oligomeric_count']
    except Exception as e:
        print(pdb, e)
        return 0



if __name__ == "__main__":
    download_proteins()


