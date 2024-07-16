from pathlib import Path
import os
import urllib.request

from Bio import Seq, SeqIO, SeqRecord
import numpy as np
import pandas as pd

import correct_index


def download_fasta(ID):
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query=accession:{ID}&format=fasta"
        print(f"Making request at:\n{url}")
        with urllib.request.urlopen(url) as f:
            fasta = f.read().decode('utf-8').strip()

        with open(f'fasta/{ID}.fasta', 'w') as o:
            o.write(fasta)
    except Exception as e:
        print(f"Error for {ID}\n{e}")


def mutate_sequence(seq, mut):
    old = mut[0]
    new = mut[-1]
    idx = int(mut[1:-1]) - 1
    if seq[idx] != old:
        print(f"WARNING! Old sequence, {seq[idx]}, does not match mutation string {mut}:\n{seq}")
    return seq[:idx] + new + seq[idx + 1:]


#####################################################
### ThermoMutDB


def load_thermomutdb(refresh=False):
    path_out = Path('thermomutdb_corrected.csv')
    if path_out.exists() and not refresh:
        return pd.read_csv(path_out)

    df = pd.read_csv('thermomutdb.csv')

    # Removing a reference that was found to have multiple extra mutations
    # compared to the WT sequence given by Uniprot
    df = df.loc[df.DOI!="10.1016/s0969-2126(00)80023-4"]

    # Correcting a single error
    df.loc[1104, 'ddg'] = -4.2

    # Changing the sign on all values of a reference that were incorrectly
    # documented
    df.loc[df.DOI=="10.1021/bi00006a025", 'ddg'] = -df.loc[df.DOI=="10.1021/bi00006a025", 'ddg']

    # Reference "10.1021/acs.jpcb.7b12121" also appears to have ddG with the wrong sign.
    # Upon further inspection, it becomes clear that the real issue is that the 'reference' dG
    # value is not the WT protein, but rather a double-mutant (\Delta + PHS).
    # There are also several duplicates, since the values reported in this reference
    # were taken from another paper.
#   df = df.loc[df.DOI!="10.1021/acs.jpcb.7b12121"]
    doi_list = ["10.1021/acs.jpcb.7b12121", "10.1073/pnas.0805113105", "10.1073/pnas.1004213107"]
    df = df.DOI.apply(lambda x: x not in doi_list)

    print(f"Removing first round of entries (pH, T, etc.)\n{len(df)} entries to start with")
    df = df.loc[(df.ph>=5)&(df.ph<=8)&(df.temperature>=293)&(df.temperature<=313)&(df.mutation_type=='Single')&(df.uniprot!='-')]
    df = df.loc[df.uniprot!="GQ884175"]
    df = df.loc[df.mutation_code!="L66LA"]
    df = df.loc[df.mutation_based!='unsigned']

    print(f"Removing by length\n{len(df)} entries left so far")
    len_key = {ID: len(list(SeqIO.parse(f"fasta/{ID}.fasta", 'fasta'))[0].seq) for ID in df.uniprot}
    df['length'] = df.uniprot.map(len_key)
    df = df.loc[(df.length>=50)&(df.length<=500)]
    
    print(f"Fixing uniprot mutation codes and removing ambiguous entries\n{len(df)} entries left")
    df = correct_index.convert_mutation_code_to_uniprot(df)

    print(f"Noting oligomers\n{len(df)} entries left")
    pdb_path = sorted(Path('../PDB/').glob("*cif"))
    oligo = {path.stem: correct_index.get_pdb_oligomer(path) for path in pdb_path}
    is_oligo = {p: False if len(o) > 1 else int(o[0]) == 1 for p, o in oligo.items()}
    idx = df.PDB_wild.notnull()
    df.loc[idx, 'is_oligo'] = df.loc[idx, 'PDB_wild'].apply(lambda x: is_oligo[x.lower()])

    print(f"Final check that sequences and mutation codes agree\n{len(df)} entries left")
    seq_key = {ID: str(list(SeqIO.parse(f"fasta/{ID}.fasta", 'fasta'))[0].seq) for ID in df.uniprot}
    df = df.loc[[seq_key[u][int(i)-1] == m[0] for u, i, m in zip(df.uniprot, df.mut_idx_uniprot, df.mutation_code)]]

    print(f"Finished!\n{len(df)} entries left")
    df.to_csv(path_out, index=False)

    return df


def load_sequences_from_uniprot_thermomutdb():
    df = load_thermomutdb()
    for ID in df.uniprot.unique():
        if not os.path.exists(f'fasta/{ID}.fasta'):
            download_fasta(ID)


def get_variant_sequences_thermomutdb():
    df = load_thermomutdb()
    df = df.loc[df.ddg.notnull()]
    df['mutation_code'] = [f"{m1[0]}{int(m2)}{m1[-1]}" for m1, m2 in zip(df.mutation_code, df.mut_idx_uniprot)]

    df = df.rename(columns={'ddg':'ddG', 'mutation_code':'mutation', 'id':'TMDB_ID', 'PDB_wild':'pdb_wt', 'pdb_mutant':'pdb_mut'})
    df['id'] = [f"{a}_{b}" for a, b in zip(df.uniprot, df.mutation)]
    cols = ['id', 'TMDB_ID', 'uniprot', 'mutation', 'is_oligo', 'pdb_wt', 'pdb_mut', 'length', 'ddG']
    df['ddG'] = -df['ddG']

    id_uniq = df.id.unique()
    rows = []
    for ID in id_uniq:
        idx = df.index[df.id==ID]
        if len(idx) > 1:
            ddg_mn = np.mean(df.loc[idx, 'ddG'])
            ddg_sd = np.std(df.loc[idx, 'ddG'])
            if ddg_sd > 1.0:
                print(ID, ddg_sd)
            else:
                rows.append(list(df.loc[idx[0], cols[:-1]].values) + [ddg_mn])
        else:
            rows.append(list(df.loc[idx[0], cols].values))

    df = pd.DataFrame(data=rows, columns=cols)
    df.to_csv('thermomutdb_processed.csv', index=False)

    seq_key = {ID: str(list(SeqIO.parse(f"fasta/{ID}.fasta", 'fasta'))[0].seq) for ID in df.uniprot}
    for wt, mut, var in zip(df.uniprot, df['mutation'], df['id']):
        mut_seq = mutate_sequence(seq_key[wt], mut)
        seq_key[var] = mut_seq
        records = [SeqRecord.SeqRecord(Seq.Seq(mut_seq), id=var, description='')] 
        path_fasta = Path(f"../fasta/{var}.fasta")
        if path_fasta.exists():
            seq_old = str(list(SeqIO.parse(path_fasta, 'fasta'))[0].seq)
            if seq_old != mut_seq:
                print(wt, mut, var)
        else:
            print(wt, mut, var)
            SeqIO.write(records, f"fasta/{var}.fasta", "fasta")
            

    records = [SeqRecord.SeqRecord(Seq.Seq(v), id=k, description='') for k, v in seq_key.items()]
    SeqIO.write(records, "fasta/all_seq.fasta", "fasta")


def run_thermomutdb():
#   load_sequences_from_uniprot_thermomutdb()
    get_variant_sequences_thermomutdb()


if __name__ == "__main__":
    run_thermomutdb()



