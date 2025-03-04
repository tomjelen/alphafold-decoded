import torch
from torch import nn

class InputEmbedder(nn.Module):
    """
    Implements Algorithm 3 and Algorithm 4.
    """

    def __init__(self, c_m, c_z, tf_dim, msa_feat_dim=49, vbins=32):
        """
        Initializes the InputEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            tf_dim (int): Embedding dimension of target_feat.
            msa_feat_dim (int, optional): Embedding dimension of the MSA feature.
                Defaults to 49.
            vbins (int, optional): Determines the bins for relpos as
                (-vbins, -vbins+1,...,vbins). Defaults to 32.
        """
        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins

        ##########################################################################
        # TODO: Initialize the modules linear_tf_z_i, linear_tf_z_j,             #
        #   linear_tf_m, linear_msa_m (from Algorithm 3) and linear_rel_pos      #
        #   (from Algorithm 4).                                                  #
        #   Note the difference between the MSA feature                          #
        #   (as created during feature extraction) and the MSA representation m  #
        #   that is used throughout the Evoformer.                               #
        ##########################################################################


        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_feat_dim, c_m)

        actual_vbins = 2 * vbins + 1 # (-vbins, -vbins+1, ..., 0, ..., vbins-1, vbins)
        self.linear_relpos = nn.Linear(actual_vbins, c_z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def relpos(self, residue_index:torch.Tensor):
        """
        Implements Algorithm 4.

        Args:
            residue_index (torch.tensor): Index of the residue in the original amino
                acid sequence. In this context, this is simply [0,... N_res-1].

        Returns:
            tuple: Tuple consisting of the embedded MSA representation m and
                pair representation z.
        """

        out = None
        dtype = self.linear_relpos.weight.dtype

        ##########################################################################
        # TODO: Implement Algorithm 4. Since the residue index is just a number, #
        #   we can directly use the shifted d_ij as class labels.                #
        #   You can follow these steps:                                          #
        #   * Cast residue_index to long.                                        #
        #   * unsqueeze residue_index accordingly to calculate the outer         #
        #      difference d_ij.                                                  #
        #   * use torch.clamp to clamp d_ij between -self.vbins and self.vbins.  #
        #   * offset the clamped d_ij by self.vbins, so that it is in the range  #
        #      [0, 2*vbins] instead of [-vbins, vbins].                          #
        #   * use nn.functional.one_hot to convert the class labels into         #
        #      one-hot encodings.                                                #
        #   * use the linear module to create the output embedding.              #
        ##########################################################################

        actual_vbins = 2 * self.vbins + 1
        residue_index = residue_index.long()

        # torch.tensor([[0],  -   torch.tensor([[0, 1, 2, 3]])   =   tensor([[ 0, -1, -2, -3],
        #               [1],                                                 [ 1,  0, -1, -2],
        #               [2],                                                 [ 2,  1,  0, -1],
        #               [3]])                                                [ 3,  2,  1,  0]])
        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2) #
        d = d.clamp(min=-self.vbins, max=self.vbins) + self.vbins

        d = nn.functional.one_hot(d, num_classes=actual_vbins)
        d = d.to(dtype)

        out = self.linear_relpos(d)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


    def forward(self, batch):
        """
        Implements the forward pass for Algorithm 3.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * msa_feat: Initial MSA feature of shape (*, N_seq, N_res, msa_feat_dim).
                * target_feat: Target feature of shape (*, N_res, tf_dim).
                * residue_index: Residue index of shape (*, N_res)

        Returns:
            tuple: Tuple consisting of the MSA representation m and the pair representation z.
        """

        m = None
        z = None

        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 3. For the calculation  #
        #   of the outer sum in line 2, the embeddings a and b must be           #
        #   unsqueezed correctly to allow for broadcasting along the N_res dim.  #
        #   Note: For batched use, target_feat must be unsqueezed after the      #
        #   computation of a and b and before the computation of m, to match     #
        #   the number of dimensions of msa_feat.                                #
        ##########################################################################

        # Linear projections of the target features
        a = self.linear_tf_z_i(target_feat)                             # (*, N_res, c_z)
        b = self.linear_tf_z_j(target_feat)                             # (*, N_res, c_z)

        # Create pair representation through outer sum:
        # 1. unsqueeze(-2): Add dimension for residue j, creating shape   (*, N_res, 1,     c_z)
        # 2. unsqueeze(-3): Add dimension for residue i, creating shape   (*, 1,     N_res, c_z)
        # 3. When added, broadcasting creates all-vs-all pairs with shape (*, N_res, N_res, c_z)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        # Add relative positional encoding
        z = z + self.relpos(residue_index)

        # Create MSA representation:
        # unsqueeze(-3): Add sequence dimension to target_feat, making it shape (*, 1,     N_res, tf_dim)
        # This allows broadcasting when adding to msa_feat which has shape      (*, N_seq, N_res, msa_feat_dim)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat.unsqueeze(-3))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return m, z
