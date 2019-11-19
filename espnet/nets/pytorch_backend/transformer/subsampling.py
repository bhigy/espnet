import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def introspect(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        activations = {}
        x = x.unsqueeze(1)  # (b, c, t, f)
        for i, module in enumerate(self.conv._modules.values()):
            x = module(x)
            if type(module) == torch.nn.ReLU:
                b, c, t, f = x.size()
                ac = x.transpose(1, 2).contiguous().view(b, t, c * f)
                activations['conv' + str(i)] = ac.cpu().numpy()
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        activations['out'] = x.cpu().numpy()
        if x_mask is None:
            return x, None, activations
        return x, x_mask[:, :, :-2:2][:, :, :-2:2], activations
