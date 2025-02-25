import torch
from torch import nn
from evoformer.dropout import DropoutRowwise
from evoformer.msa_stack import MSARowAttentionWithPairBias, MSAColumnAttention, OuterProductMean, MSATransition
from evoformer.pair_stack import PairStack


class EvoformerBlock(nn.Module):
    """
    Implements one block from Algorithm 6.
    """

    def __init__(self, c_m, c_z):
        """Initializes EvoformerBlock.

        Args:
            c_m (int): Embedding dimension for the MSA representation.
            c_z (int): Embedding dimension for the pair representation.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules msa_att_row, msa_att_col, msa_transition, #
        #   outer_product_mean, core (the PairStack), and (optionally for        #
        #   inference) dropout_rowwise_m.                                        #
        ##########################################################################

        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z)
        self.msa_att_col = MSAColumnAttention(c_m)
        self.msa_transition = MSATransition(c_m)
        self.outer_product_mean = OuterProductMean(c_m, c_z)
        self.core = PairStack(c_z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m, z):
        """
        Implements the forward pass for one block in Algorithm 6.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            tuple: Transformed tensors m and z of the same shape as the inputs.
        """

        ##########################################################################
        # TODO: Implement  the forward pass for Algorithm 6.                     #
        ##########################################################################

        # MSA stack
        m = m + self.msa_att_row(m, z)
        m = m + self.msa_att_col(m)
        m = m + self.msa_transition(m)

        # Communication
        z = z + self.outer_product_mean(m)

        # Pair stack
        z = self.core(z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return m, z

class EvoformerStack(nn.Module):
    """
    Implements Algorithm 6.
    """

    def __init__(self, c_m, c_z, num_blocks, c_s=384):
        """
        Initializes the EvoformerStack.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            num_blocks (int): Number of blocks for the Evoformer.
            c_s (int, optional): Number of channels for the single representation.
                Defaults to 384.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize self.blocks as a ModuleList of EvoformerBlocks        #
        #   and self.linear as the extraction of the single representation.      #
        ##########################################################################

        self.blocks = nn.ModuleList([EvoformerBlock(c_m, c_z) for _ in range(num_blocks)])
        self.linear = nn.Linear(c_m, c_s)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m, z):
        """
        Implements the forward pass for Algorithm 6.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            tuple: Output tensors m, z, and s, where m and z have the same shape
                as the inputs and s has shape (*, N_res, c_s)
        """

        s = None

        ##########################################################################
        # TODO: Implement  the forward pass for Algorithm 6.                     #
        #   The single representation is created by embedding the first row      #
        #   of the msa representation.                                           #
        ##########################################################################

        for _, evo_block in enumerate(self.blocks):
            m, z = evo_block(m, z)

        single_representation = m[..., 0, :, :] # MSA representation for the first sequence (target sequence)
        s = self.linear(single_representation)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return m, z, s
