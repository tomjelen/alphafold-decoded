import torch
import math
from torch import nn

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        Initializes the module. MultiHeadAttention theoretically consists of
        N_head separate linear layers for the query, key and value embeddings.
        However, the embeddings can be computed jointly and split afterwards,
        so we only need one query, key and value layer with larger c_out.

        Args:
            c_in (int): Input dimension for the embeddings.
            c (int): Embedding dimension for each individual head.
            N_head (int): Number of heads.
            attn_dim (int): The dimension in the input tensor along which
                the attention mechanism is performed.
            gated (bool, optional): If True, an additional sigmoid-activated
                linear layer will be multiplicated against the weighted
                value vectors before feeding them through the output layer.
                Defaults to False.
            is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query,
                key, and value embeddings will use bias, otherwise not.
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        ##########################################################################
        # TODO: Initialize the query, key, value and output layers.              #
        #   Whether or not query, key, and value layers use bias is determined   #
        #   by `use_bias` (False for AlphaFold). The output layer should always  #
        #   use a bias. If gated is true, initialize another linear with bias.   #
        #   For compatibility use the names linear_q, linear_k, linear_v,        #
        #   linear_o and linear_g.                                               #
        ##########################################################################

        if is_global:
            self.linear_q = torch.nn.Linear(c_in, c*N_head, use_bias_for_embeddings)
            self.linear_k = torch.nn.Linear(c_in, c, use_bias_for_embeddings)
            self.linear_v = torch.nn.Linear(c_in, c, use_bias_for_embeddings)
            self.linear_o = torch.nn.Linear(c*N_head, c_in, True)
        else:
            self.linear_q = torch.nn.Linear(c_in, c*N_head, use_bias_for_embeddings)
            self.linear_k = torch.nn.Linear(c_in, c*N_head, use_bias_for_embeddings)
            self.linear_v = torch.nn.Linear(c_in, c*N_head, use_bias_for_embeddings)
            self.linear_o = torch.nn.Linear(c*N_head, c_in, True)

        if gated:
            self.linear_g = torch.nn.Linear(c_in, c*N_head, True)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Splits the embeddings into individual heads and transforms the input
        shapes of form (*, q/k/v, *, N_head*c) into the shape
        (*, N_head, q/k/v, c). The position of the q/k/v dimension
        in the original tensors is given by attn_dim.

        Args:
            q (torch.Tensor): Query embedding of shape (*, q, *, N_head*c).
            k (torch.Tensor): Key embedding of shape (*, k, *, N_head*c).
            v (torch.Tensor): Value embedding of shape (*, v, *, N_head*c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, q/k/v, c) respectively.
        """

        ##########################################################################
        # TODO: Rearrange the tensors with the following changes:                #
        #   - (*, q/k/v, *, N_head*c) -> (*, q/k/v, N_head*c) with movedim       #
        #   - (*, q/k/v, N_head*c) -> (*, q/k/v, N_head, c)                      #
        #   - (*, q/k/v, N_head, c) -> (*, N_head, q/k/v, c)                     #
        ##########################################################################

        q = q.movedim(self.attn_dim, -2)
        q = q.reshape(*q.shape[:-1], self.N_head, self.c)
        q = q.movedim(-2, -3)

        k = k.movedim(self.attn_dim, -2)
        k = k.reshape(*k.shape[:-1], self.N_head, self.c)
        k = k.movedim(-2, -3)

        v = v.movedim(self.attn_dim, -2)
        v = v.reshape(*v.shape[:-1], self.N_head, self.c)
        v = v.movedim(-2, -3)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return q, k, v

    def prepare_qkv_global(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        """
        Prepares the query, key and value embeddings with the following
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.


        Args:
            q (torch.tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (torch.tensor): Key embeddings of shape (*, k, *, c).
            v (torch.tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v.
        """

        ##########################################################################
        # TODO: Rearrange the tensors to match the output dimensions. Use        #
        #   torch.mean for the contraction of q at the end of this function.     #
        ##########################################################################

        q = q.movedim(self.attn_dim, -2)                    # (*, q, N_head*c)
        q = q.reshape(*q.shape[:-1], self.N_head, self.c)   # (*, q, N_head, c)
        q = q.movedim(-2, -3)                               # (*, N_head, q, c)
        q = q.mean(dim=-2, keepdim=True)                    # (*, N_head, 1, c)

        k = k.movedim(self.attn_dim, -2)                    # (*, k/v, c)
        k = k.reshape(*k.shape[:-1], 1, self.c)             # (*, k/v, 1, c)
        k = k.movedim(-2, -3)                               # (*, 1, k/v, c)

        v = v.movedim(self.attn_dim, -2)                    # (*, k/v, c)
        v = v.reshape(*v.shape[:-1], 1, self.c)             # (*, k/v, 1, c)
        v = v.movedim(-2, -3)                               # (*, 1, k/v, c)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        """
        Forward pass through the MultiHeadAttention module.

        Args:
            x (torch.tensor): Input tensor of shape (*, q/k/v, *, c_in).
            bias (torch.tensor, optional): Optional bias tensor of shape
                (*, N_head, q, k) that will be added to the attention weights.
                Defaults to None.
            attention_mask (torch.tensor, optional): Optional attention mask
                of shape (*, k). If set, the keys with value 0 in the mask will
                not be attended to.

        Returns:
            torch.tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass consisting of the following steps:    #
        #   - Create query, key and value embeddings.                            #
        #   - Rearrange the embeddings with prepare_qkv.                         #
        #   - Scale the queries by 1/sqrt(c).                                    #
        #   - Calculate the attention weights of shape (*, N_head, q, k)         #
        #       from q and k. You can use torch.einsum for this.                 #
        #   - If a bias was given:                                               #
        #       - extract the bias batch shape by omitting the last 3 dims       #
        #         from bias.                                                     #
        #       - construct a broadcastable bias shape, by concatenating         #
        #           bias_batch_shape, (1,) * n, and the last three dims of bias. #
        #           Choose n such that the broadcastable shape has as many dims  #
        #           as the attention scores.                                     #
        #       - add the bias to the attention scores.                          #
        #   - If an attention mask was given (not needed for AlphaFold):         #
        #       - unsqueeze the mask to make it broadcastable against the        #
        #         attention scores of shape (*, N_head, q, k).                   #
        #       - create a tensor `offset`` of the same shape as the mask with   #
        #         the value -1e8 where the mask is 0 and zero elsewhere.         #
        #       - add the offset to the raw attention scores.                    #
        #   - Use softmax to convert the attention scores into a                 #
        #       probability distribution.                                        #
        #   - Weight the value vectors by the attention weights and sum          #
        #       them up along the key dimension. You can use torch.einsum        #
        #       to do this in one line. The result should be                     #
        #       of shape (*, N_head, q, c).                                      #
        #   - Rearrange the intermediate output in the following way:            #
        #       * (*, N_head, q, c) -> (*, q, N_head, c)                         #
        #       * (*, q, N_head, c) -> (*, q, N_head * c)                        #
        #       * (*, q, N_head * c) -> (*, q, *, N_head * c)                    #
        #       The order of these transformations is crucial, as moving q       #
        #       to attn_dim before flattening the heads will result in an        #
        #       incorrect positioning if attn_dim uses negative indexing.        #
        #   - if gated, calculate the gating with linear_g and sigmoid and       #
        #       multiply it against the output.                                  #
        #   - apply linear_o to calculate the final output.                        #
        ##########################################################################

        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        if self.is_global:
            q, k, v = self.prepare_qkv_global(q,k,v) # (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v
        else:
            q, k, v = self.prepare_qkv(q,k,v) # (*, N_head, q/k/v, c)

        q /= math.sqrt(self.c)

        attn_weights = torch.einsum("...Nqc,...Nkc->...Nqk", q, k) # (*, N_head, q, k)

        # bias = (*, N_head, q, k)
        if bias is not None:
            # Reshape to (<BiasBatchShape>, <Ones>, <BiasShape>) to match attn_weights.shape with 1's in the middle so that we broadcast here
            # Example:
            # - bias (*, N_head, q, k) is:                    (2, 4, 6, 6)
            # - attn_weights (*, N_head, q, k) is:   (2, 3, 5, 7, 4, 6, 6)
            # - desired shape is:                    (2, 1, 1, 1, 4, 6, 6)
            bias_batch_shape = bias.shape[:-3]
            n_ones = len(attn_weights.shape) - len(bias_batch_shape) - 3
            ones = (1,) * n_ones
            broadcastable_shape = bias_batch_shape + ones + bias.shape[-3:]
            bias = bias.reshape(broadcastable_shape)

            attn_weights = attn_weights + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            offset = (attention_mask==0) * -1e8
            attn_weights = attn_weights + offset

        scores = attn_weights.softmax(-1)
        attn = scores @ v # (*, N_head, q, c)

        attn = attn.movedim(-3, -2)               # (*, N_head, q, c)  -> (*, q, N_head, c)
        attn = attn.flatten(-2)                   # (*, q, N_head, c)  -> (*, q, N_head * c)
        attn = attn.movedim(-2, self.attn_dim)    # (*, q, N_head * c) -> (*, q, *, N_head * c)

        out = self.linear_o(attn)

        if self.gated:
            out = torch.sigmoid(self.linear_g(out)) * out

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
