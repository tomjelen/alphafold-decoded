import torch
from torch import nn
from evoformer.dropout import DropoutRowwise, DropoutColumnwise
from attention.mha import MultiHeadAttention


class TriangleMultiplication(nn.Module):
    """
    Implements Algorithm 11 and Algorithm 12.
    """

    def __init__(self, c_z, mult_type, c=128):
        """Initialization of TriangleMultiplication.

        Args:
            c_z (int): Embedding dimension of the pair representation.
            mult_type (str): Either 'outgoing' for Algorithm 11 or 'incoming' for Algorithm 12.
            c (int, optional): Embedding dimension . Defaults to 128.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if mult_type not in {'outgoing', 'incoming'}:
            raise ValueError(f'mult_type must be either "outgoing" or "incoming" but is {mult_type}.')

        self.mult_type = mult_type

        ##########################################################################
        # TODO: Initialize the modules layer_norm_in, layer_norm_out,            #
        #   linear_a_p, linear_a_g, linear_b_p, linear_b_g, linear_g and         #
        #   linear_z. The layers linear_a_g, linear_b_g and linear_g are the     #
        #   gating layers, which are passed through sigmoid and multiplied by    #
        #   the values, which in turn are being created by linear_a_p,           #
        #   linear_b_p and linear_z.                                             #
        ##########################################################################

        self.layer_norm_in = nn.LayerNorm(c_z)

        self.linear_a_p = nn.Linear(c_z, c)
        self.linear_a_g = nn.Linear(c_z, c)

        self.linear_b_p = nn.Linear(c_z, c)
        self.linear_b_g = nn.Linear(c_z, c)

        self.linear_z = nn.Linear(c, c_z)
        self.linear_g = nn.Linear(c_z, c_z)

        self.layer_norm_out = nn.LayerNorm(c)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, z):
        """
        Implements the forward pass for Algorithm 11 or Algorithm 12, depending
        on if mult_type is set to 'outgoing' or 'incoming'.

        Args:
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the  same shape as z.
        """
        out = None
        ##########################################################################
        # TODO: Use the modules initialized in __init__ to implement Algorithm   #
        #   11 or Algorithm 12, depending on self.mult_type. The only difference #
        #   lies in line 4 of the algorithm: For outgoing multiplication,        #
        #   the different rows are broadcasted in an each-with-each fasion,      #
        #   while the columns are contracted. For incoming multiplication,       #
        #   it is the other way around. You can implement this efficiently       #
        #   using torch.einsum.                                                  #
        ##########################################################################

        z = self.layer_norm_in(z)                                   # (*, N_res, N_res, c_z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)  # (*, N_res, N_res, c)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)  # (*, N_res, N_res, c)
        g = torch.sigmoid(self.linear_g(z))                         # (*, N_res, N_res, c_z)

        if self.mult_type == "outgoing":
            # Alg 11
            triangle_sum = torch.einsum('...ikc,...jkc->...ijc', a, b)
        else:
            # Alg 12
            triangle_sum = torch.einsum('...kic,...kjc->...ijc', a, b)

        # Output projection with gating
        z = self.layer_norm_out(triangle_sum)
        z = self.linear_z(z)
        out = g * z

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class TriangleAttention(nn.Module):
    """
    Implements Algorithm 13 and Algorithm 14.
    """
    def __init__(self, c_z, node_type, c=32, N_head=4):
        """
        Initialization of TriangleAttention.

        Args:
            c_z (int): Embedding dimension of the pair representation.
            node_type (str): Either 'starting_node' for Algorithm 13 or 'ending_node' for Algorithm 14.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 4.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        if node_type not in {'starting_node', 'ending_node'}:
            raise ValueError(f'node_type must be either "starting_node" or "ending_node" but is {node_type}')

        self.node_type = node_type

        ##########################################################################
        # TODO: Initialize the modules layer_norm, mha and linear.               #
        #   The module linear is used to embed the bias.                         #
        #   Note that the attention dimension for MultiHeadAttention             #
        #   depends on node_type. For attention around the starting node,        #
        #   attention is broadcasted over the rows and the attention mechanism   #
        #   is calculated over the different columns. For attention around the   #
        #   ending node, it is the other way around.                             #
        ##########################################################################

        self.layer_norm = nn.LayerNorm(c_z)
        self.mha = MultiHeadAttention(
            c_in=c_z,
            c=c,
            N_head=N_head,
            attn_dim=-2 if node_type == 'starting_node' else -3,
            gated=True,
        )
        self.linear = nn.Linear(c_z, N_head, bias=False)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, z:torch.Tensor):
        """
        Implements the forward pass for Algorithm 13 or Algorithm 14, depending
        if node_type is set to 'starting_node' or 'ending_node'.

        Args:
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        out = None

        ##########################################################################
        # TODO: Use the modules initialized in __init__ to implement Algorithm   #
        #   13 or Algorithm 14, depending on self.node_type.                     #
        #   The different attention dimension is already set in the init method. #
        #   Still, the implementations differ: For attention around the ending   #
        #   node, the attention score of how much value k is contributing        #
        #   to query i is determined by the incoming bias b_ki instead of the    #
        #   outgoing bias b_ik: The bias is transposed.                          #
        ##########################################################################

        z:torch.Tensor = self.layer_norm(z)             # (*, N_res, N_res, c_z)
        b:torch.Tensor = self.linear(z)                 # (*, N_res, N_res, N_head)
        b:torch.Tensor = b.movedim(-1, -3)              # (*, N_head, N_res, N_res)

        if self.node_type == "ending_node":
            b = b.transpose(-1, -2)                     # (*, N_head, N_res, N_res)

        out = self.mha(z, bias=b)                       # (*, N_res, N_res, c_z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n=4):
        """
        Initializes the PairTransition.

        Args:
            c_z (int): Embedding dimension of the pair representation.
            n (int, optional): Factor by which the number of intermediate channels
                expands the original number of channels.
                Defaults to 4.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm, linear_1, relu and linear_2.  #
        ##########################################################################

        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_1 = nn.Linear(c_z, c_z*n)
        self.linear_2 = nn.Linear(c_z*n, c_z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, z):
        """
        Forward pass for Algorithm 15.

        Args:
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 15.                     #
        ##########################################################################

        z = self.layer_norm(z)
        a = self.linear_1(z)
        out = self.linear_2(torch.relu(a))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out



class PairStack(nn.Module):
    """
    Implements the pair stack from Algorithm 6.
    """

    def __init__(self, c_z):
        """
        Initializes the PairStack.

        Args:
            c_z (int): Embedding dimension of the pair representation.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules tri_mul_out, tri_mul_in, tri_att_start,    #
        #   tri_att_end, pair_transition, and (optionally for inference)         #
        #   dropout_rowwise and dropout_columnwise.                              #
        ##########################################################################

        self.tri_mul_out = TriangleMultiplication(c_z, "outgoing")
        self.tri_mul_in = TriangleMultiplication(c_z, "incoming")
        self.tri_att_start = TriangleAttention(c_z, "starting_node")
        self.tri_att_end = TriangleAttention(c_z, "ending_node")
        self.pair_transition = PairTransition(c_z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, z):
        """
        Implements the forward pass for the pair stack from Algorithm 6.

        Args:
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass for the pair stack from Algorithm 6.  #
        ##########################################################################

        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + self.tri_att_end(z)
        out = z + self.pair_transition(z)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
