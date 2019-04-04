import torch
from torch.autograd import Function
from .._ext import corr

class correlation(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        super(correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        self.save_for_backward(input1, input2)
        
        rbot1 = input1.new()
        rbot2 = input2.new()
        output = input1.new()

        corr.corr_cuda_forward(input1, input2,
                               rbot1, rbot2,
                               output,
                               self.pad_size,
                               self.kernel_size,
                               self.max_displacement,
                               self.stride1,
                               self.stride2,
                               self.corr_multiply)

        return output

    def backward(self, grad_output):

        input1, input2 = self.saved_tensors

        rbot1 = input1.new()
        rbot2 = input2.new()

        grad_input1 = torch.zeros(input1.size()).cuda()
        grad_input2 = torch.zeros(input2.size()).cuda()

        corr.corr_cuda_backward(input1, input2,
                                rbot1, rbot2,
                                grad_output,
                                grad_input1,
                                grad_input2,
                                self.pad_size,
                                self.kernel_size,
                                self.max_displacement,
                                self.stride1,
                                self.stride2,
                                self.corr_multiply)

        return grad_input1, grad_input2


#----- 1D correlation (for disparity) Jinwei Gu -----

class correlation1d(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        super(correlation1d, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        self.save_for_backward(input1, input2)

        rbot1 = input1.new()
        rbot2 = input2.new()
        output = input1.new()

        corr.corr1d_cuda_forward(input1, input2,
                               rbot1, rbot2,
                               output,
                               self.pad_size,
                               self.kernel_size,
                               self.max_displacement,
                               self.stride1,
                               self.stride2,
                               self.corr_multiply)

        return output

    def backward(self, grad_output):

        input1, input2 = self.saved_tensors

        rbot1 = input1.new()
        rbot2 = input2.new()

        grad_input1 = torch.zeros(input1.size()).cuda()
        grad_input2 = torch.zeros(input2.size()).cuda()

        #grad_input1 = grad_output.new()
        #grad_input2 = grad_output.new()

        corr.corr1d_cuda_backward(input1, input2,
                                rbot1, rbot2,
                                grad_output,
                                grad_input1,
                                grad_input2,
                                self.pad_size,
                                self.kernel_size,
                                self.max_displacement,
                                self.stride1,
                                self.stride2,
                                self.corr_multiply)

        return grad_input1, grad_input2
