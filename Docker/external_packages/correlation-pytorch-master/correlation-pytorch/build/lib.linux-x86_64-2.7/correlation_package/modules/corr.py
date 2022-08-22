from torch.nn.modules.module import Module
from ..functions.corr import correlation, correlation1d

class Correlation(Module):

    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None,
                 stride1=None, stride2=None, corr_multiply=None):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        return correlation(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)

    def __repr__(self):
        return self.__class__.__name__


#----- correlation in 1D (for disparity) Jinwei Gu -----

class Correlation1d(Module):

    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None,
                 stride1=None, stride2=None, corr_multiply=None):
        super(Correlation1d, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        return correlation1d(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)

    def __repr__(self):
        return self.__class__.__name__
