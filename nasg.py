import torch
import transform

NUM_NASG_PARAMETERS = 8

class NASG(transform.Transform):
    """
    """

    def __init__(self, num_mixtures: int, transform_net_create_fn, blob: int = None, num_context_features: int=0):
        """
        Constructor.
        Args:
            num_mixtures : int for number of NASG mixtures
            transform_net_create_fn : lambda defining a network based on in_features and out_features
            blob : int for number of bins to include in the input one-blob encoding
            num_context_features: number of context features in transform net
        """
        super().__init__()
        if not isinstance(num_mixtures, int) and num_mixtures > 0:
            raise ValueError('num_mixtures requires a number of mixtures > 0')
        self.num_mixtures = num_mixtures
        self.num_context_features = num_context_features

        self.blob = bool(blob)
        if self.blob:
            if not isinstance(blob, int):
                raise ValueError('Blob encoding requires a number of bins')
            self.nbins_in = int(blob)

        if self.blob:
            self.transform_net = transform_net_create_fn(
                self.num_context_features * self.nbins_in,
                self.num_mixtures * NUM_NASG_PARAMETERS + 1
            )
        else:
            self.transform_net = transform_net_create_fn(
                self.num_context_features,
                self.num_mixtures * NUM_NASG_PARAMETERS + 1
            )

    def one_blob(self,xd):
        device = xd.get_device() if xd.is_cuda else torch.device('cpu')
        binning = (0.5/self.nbins_in) + torch.arange(0., 1.,1./self.nbins_in, device=device).repeat(xd.numel())
        binning = binning.reshape(-1,self.num_context_features,self.nbins_in)
        x = xd.unsqueeze(-1)
        res = torch.exp(((-self.nbins_in*self.nbins_in)/2.) * (binning-x)**2)
        return res.reshape(-1,self.num_context_features*self.nbins_in)

    def forward(self, inputs, context=None):
        """
        inputs : uniform x,y for NASG sampling
        context : features for network
        """

        if context.shape[1] != self.num_context_features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.num_context_features, inputs.shape[1]))

        if self.blob:
            context = self.one_blob(context)
        nasg_params = self.transform_net(context)  # TODO extract 'c' coefficitent from output
        outputs, absdet = self._nasg_transform_forward(
            inputs=inputs,
            nasg_params=nasg_params
        )
        return outputs, absdet

    def inverse(self, inputs, context=None):
        """
        inputs : uniform x,y for NASG sampling
        context : features for network
        """
        if context.shape[1] != self.num_context_features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.num_context_features, inputs.shape[1]))

        if self.blob:
            context = self.one_blob(context)

        nasg_params = self.transform_net(context)  # TODO extract 'c' coefficitent from output

        outputs, absdet = self._nasg_transform_inverse(
            inputs=inputs,
            nasg_params=nasg_params
        )
        return outputs, absdet

    def _nasg_transform_forward(self, inputs, nasg_params):
        """
        inputs : uniform x,y for NASG sampling
        nasg_params : NASG parameter at the context
        """
        raise NotImplementedError()

    def _nasg_transform_inverse(self, inputs, nasg_params):
        """
        inputs : uniform x,y for NASG sampling
        context : NASG parameter at the context
        """
        raise NotImplementedError()