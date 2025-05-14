#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import sqrt

from Bio import pairwise2
from Bio.Align import substitution_matrices, PairwiseAligner


def aar(candidate, reference):
    hit = 0
    for a, b in zip(candidate, reference):
        if a == b:
            hit += 1
    return hit / len(reference)


# def align_sequences(sequence_A, sequence_B, **kwargs):
#     """
#     Performs a global pairwise alignment between two sequences
#     using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
#     as implemented in Biopython. Returns the alignment, the sequence
#     identity and the residue mapping between both original sequences.
#     """
# 
#     sub_matrice = substitution_matrices.load('BLOSUM62')
#     aligner = PairwiseAligner()
#     aligner.substitution_matrix = sub_matrice
#     alns = aligner.align(sequence_A, sequence_B)
# 
#     best_aln = alns[0]
#     aligned_A, aligned_B = best_aln
# 
#     base = sqrt(aligner.score(sequence_A, sequence_A) * aligner.score(sequence_B, sequence_B))
#     seq_id = aligner.score(sequence_A, sequence_B) / base
#     return (aligned_A, aligned_B), seq_id


def align_sequences(sequence_A, sequence_B, symmetric=False, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.

    The choices of gap_open and gap_extend are domain conventions which
    relate to the usage of BLOSUM62
    """

    def _calculate_identity(sequenceA, sequenceB):
        """
        Returns the percentage of identical characters between two sequences.
        Assumes the sequences are aligned.
        """

        sa, sb, sl = sequenceA, sequenceB, len(sequenceA)
        matches = [sa[i] == sb[i] for i in range(sl)]
        seq_id = sum(matches) / sl
        return seq_id

    matrix = kwargs.get('matrix', substitution_matrices.load("BLOSUM62"))
    gap_open = kwargs.get('gap_open', -10.0)
    gap_extend = kwargs.get('gap_extend', -0.5)

    alns = pairwise2.align.globalds(sequence_A, sequence_B,
                                    matrix, gap_open, gap_extend,
                                    penalize_end_gaps=(False, False) )

    best_aln = alns[0]
    aligned_A, aligned_B, score, begin, end = best_aln

    # Calculate sequence identity
    seq_id = _calculate_identity(aligned_A, aligned_B)

    if symmetric:
        alns = pairwise2.align.globalds(sequence_B, sequence_A,
                                        matrix, gap_open, gap_extend,
                                        penalize_end_gaps=(False, False) )

        best_aln = alns[0]
        aligned_B, aligned_A, score, begin, end = best_aln
        seq_id_reverse = _calculate_identity(aligned_B, aligned_A)
        seq_id = (seq_id + seq_id_reverse) * 0.5

    return (aligned_A, aligned_B), seq_id


def slide_aar(candidate, reference, aar_func=aar):
    '''
    e.g.
     candidate: AILPV
     reference: ILPVH

     should be matched as
     AILPV
      ILPVH

    To do this, we slide the candidate and calculate the maximum aar:
        A
       AI
      AIL
     AILP
    AILPV
    ILPV 
    LPV  
    PV   
    V    
    '''
    special_token = ' '
    ref_len = len(reference)
    padded_candidate = special_token * (ref_len - 1) + candidate + special_token * (ref_len - 1)
    value = 0
    for start in range(len(padded_candidate) - ref_len + 1):
        value = max(value, aar_func(padded_candidate[start:start + ref_len], reference))
    return value


if __name__ == '__main__':
    print(align_sequences('PKGYAAPSA', 'KPAVYKFTL'))
    print(align_sequences('KPAVYKFTL', 'PKGYAAPSA'))
    print(align_sequences('PKGYAAPSA', 'PKGYAAPSA'))
    print(align_sequences('KPAVYKFTL', 'KPAVYKFTL'))
