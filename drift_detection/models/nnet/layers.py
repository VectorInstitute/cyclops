import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

"""
Subclasses of torch.nn.Module
"""


class FeedforwardNet(torch.nn.Module):
    """
    Feedforward network of arbitrary size
    """

    def __init__(
        self,
        in_features,
        hidden_dim_list=[],
        output_dim=2,
        drop_prob=0.0,
        normalize=False,
        activation=F.leaky_relu,
        sparse=False,
        sparse_mode="csr",
        resnet=False,
        spectral_norm=False,
        include_output_layer=True,
        add_revgrad_to_start=False, # add gradient reversal layer to the beginning
        add_revgrad_to_end=False, # add gradient reversal to the end
        revgrad_lambd = 1. #scaling parameter for gradient reversal layer
    ):
        super().__init__()

        num_hidden = len(hidden_dim_list)
        ## If no hidden layers - go right from input to output (equivalent to logistic regression)
        if num_hidden == 0:
            output_layer = LinearLayerWrapper(
                in_features,
                output_dim,
                sparse=sparse,
                sparse_mode=sparse_mode,
                spectral_norm=spectral_norm,
            )
            self.layers = nn.ModuleList([output_layer])

        ## If 1 or more hidden layer, create input and output layer separately
        elif num_hidden >= 1:
            input_layer = HiddenLinearLayer(
                in_features=in_features,
                out_features=hidden_dim_list[0],
                drop_prob=drop_prob,
                normalize=normalize,
                activation=activation,
                sparse=sparse,
                sparse_mode=sparse_mode,
                spectral_norm=spectral_norm,
            )
            self.layers = nn.ModuleList([input_layer])
            if resnet:
                self.layers.extend(
                    [
                        ResidualBlock(
                            hidden_dim=hidden_dim_list[0],
                            drop_prob=drop_prob,
                            normalize=normalize,
                            activation=activation,
                            spectral_norm=spectral_norm,
                        )
                    ]
                )

            output_layer = LinearLayerWrapper(
                hidden_dim_list[-1],
                output_dim,
                sparse=False,
                spectral_norm=spectral_norm,
            )

            ## If more than one hidden layer, create intermediate hidden layers
            if num_hidden > 1:
                ## Standard feedforward network
                if not resnet:
                    self.layers.extend(
                        [
                            HiddenLinearLayer(
                                in_features=hidden_dim_list[i],
                                out_features=hidden_dim_list[i + 1],
                                drop_prob=drop_prob,
                                normalize=normalize,
                                activation=activation,
                                sparse=False,
                                spectral_norm=spectral_norm,
                            )
                            for i in range(num_hidden - 1)
                        ]
                    )
                else:  # Resnet-like architecture
                    for i in range(num_hidden - 1):
                        if hidden_dim_list[i] is not hidden_dim_list[i + 1]:
                            self.layers.extend(
                                [
                                    HiddenLinearLayer(
                                        in_features=hidden_dim_list[i],
                                        out_features=hidden_dim_list[i + 1],
                                        drop_prob=drop_prob,
                                        normalize=normalize,
                                        activation=activation,
                                        sparse=False,
                                        spectral_norm=spectral_norm,
                                    )
                                ]
                            )
                        self.layers.extend(
                            [
                                ResidualBlock(
                                    hidden_dim=hidden_dim_list[i + 1],
                                    drop_prob=drop_prob,
                                    normalize=normalize,
                                    activation=activation,
                                    spectral_norm=spectral_norm,
                                )
                            ]
                        )
            self.layers.extend([output_layer])
        
        if add_revgrad_to_start:
            self.layers.insert(0, RevGrad(lambd=revgrad_lambd))
        if add_revgrad_to_end:
            self.layers.extend(RevGrad(lambd=revgrad_lambd))

    def forward(self, x):
        y_pred = nn.Sequential(*self.layers).forward(x)
        return y_pred
    

class EncoderDecoderNet(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder.forward(self.encoder.forward(x))


class SequentialLayers(nn.Module):
    """
    Wraps an arbitrary list of layers with nn.Sequential.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return nn.Sequential(*self.layers).forward(x)


class LinearLayer(nn.Module):
    """
    Linear Regression model
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class SparseLinear(nn.Module):
    """
    This is a replacement for nn.Linear where the input matrix may be a sparse tensor.
    Note that the weight attribute is defined as the transpose of the definition in nn.Linear
    """

    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input):
        return self.sparse_linear(input, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def sparse_linear(self, input, weight, bias=None):
        output = torch.sparse.mm(input, weight)
        if bias is not None:
            output += bias
        ret = output
        return ret

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class EmbeddingBagLinear(nn.Module):
    """
    This is a more efficient replacement for SparseLinear that uses an EmbeddingBag
    """

    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.embed = nn.EmbeddingBag(
            num_embeddings=in_features, embedding_dim=out_features, mode="sum"
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embed.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.embed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def compute_offsets(self, batch):
        return (
            torch.LongTensor(batch.indices).to(self.embed.weight.device, non_blocking=True),
            torch.LongTensor(batch.indptr[:-1]).to(self.embed.weight.device, non_blocking=True)
            if batch.shape[0] > 1
            else torch.LongTensor([0]).to(self.embed.weight.device, non_blocking=True),
            torch.FloatTensor(batch.data).to(self.embed.weight.device, non_blocking=True),
        )

    def forward(self, x):
        if self.bias is not None:
            return self.embed(*self.compute_offsets(x)) + self.bias
        else:
            return self.embed(*self.compute_offsets(x))


class EmbeddingBagLinearDict(EmbeddingBagLinear):
    def compute_offsets(self, batch):
        return (
            batch["col_id"],
            batch["indptr"],
            batch["data"],
        )


class LinearLayerWrapper(torch.nn.Module):
    """
    Wrapper around various linear layers to call appropriate sparse layer
    """

    def __init__(
        self,
        in_features,
        out_features,
        sparse=False,
        sparse_mode="csr",
        spectral_norm=False,
    ):
        super().__init__()
        if sparse and (sparse_mode == "csr"):
            self.linear = EmbeddingBagLinear(in_features, out_features)
        elif sparse and (sparse_mode == "convert"):
            self.linear = SparseLinear(in_features, out_features)
        elif sparse and (sparse_mode == "dict"):
            self.linear = EmbeddingBagLinearDict(in_features, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)

        if spectral_norm:
            self.linear.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.linear = torch.nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        return self.linear.forward(x)


class HiddenLinearLayer(torch.nn.Module):
    """
    A neural network layer
    """

    def __init__(
        self,
        in_features,
        out_features,
        drop_prob=0.0,
        normalize=False,
        normalize_mode="batch",
        activation=F.leaky_relu,
        sparse=False,
        sparse_mode="csr",
        spectral_norm=False,
    ):
        super().__init__()

        self.linear = LinearLayerWrapper(
            in_features,
            out_features,
            sparse=sparse,
            sparse_mode=sparse_mode,
            spectral_norm=spectral_norm,
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.activation = activation
        self.normalize = normalize
        if self.normalize:
            if normalize_mode == "layer":
                self.normalize_layer = nn.LayerNorm(normalized_shape=out_features)
            elif normalize_mode == "batch":
                self.normalize_layer = nn.BatchNorm1d(out_features)
            else:
                raise ValueError('normalize_mode must be "layer" or "batch"')

    def forward(self, x):
        if self.normalize:
            result = self.dropout(self.activation(self.normalize_layer(self.linear(x))))
        else:
            result = self.dropout(self.activation(self.linear(x)))
        return result


class ResidualBlock(torch.nn.Module):
    """
    A residual block for fully connected networks
    """

    def __init__(
        self,
        hidden_dim,
        drop_prob=0.0,
        normalize=False,
        activation=F.leaky_relu,
        spectral_norm=False,
    ):
        super().__init__()

        self.layer1 = HiddenLinearLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            drop_prob=drop_prob,
            normalize=normalize,
            activation=activation,
            spectral_norm=spectral_norm,
        )
        self.layer2 = HiddenLinearLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            drop_prob=drop_prob,
            normalize=normalize,
            activation=self.identity,
            spectral_norm=spectral_norm,
        )

        self.activation = activation

    def forward(self, x):
        result = self.activation(self.layer2(self.layer1(x)) + x)
        return result

    def identity(self, x):
        return x
    
class RevGrad_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, lambd_):
        ctx.save_for_backward(input_, lambd_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        grad_input = None
        _, lambd_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * lambd_
        return grad_input, None


revgrad = RevGrad_F.apply

class RevGrad(torch.nn.Module):
    def __init__(self, lambd=1., *args, **kwargs):
        """
        A gradient reversal layer that reverses the gradient
        in the backward pass scaled by the lambd parameter.
        """
        super().__init__(*args, **kwargs)

        self._lambd = torch.tensor(lambd, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._lambd)