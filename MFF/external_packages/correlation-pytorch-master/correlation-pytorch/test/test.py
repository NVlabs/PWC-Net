import torch
import torch.nn as nn
from torch.autograd import Variable
from correlation_package.modules.corr import Correlation, Correlation1d
from correlation_package.functions.corr import correlation, correlation1d

from torch.autograd import gradcheck

import numpy as np

def test_correlation():
#    model = correlation(1, 1, 1, 1, 1, 1)
#    A = Variable(torch.randn(1,1,3,3))
#    A_ = A.cuda()
#    B = Variable(torch.randn(1,1,3,3))
#    B_ = B.cuda()
#
#   #import pdb; pdb.set_trace()
#    #model = correlation1d(3, 1, 20, 1, 1, 1)
#    y = model(A_, B_)
#    print(y.size())
#
#    print(y)
#    return


    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda()

    model = correlation(3, 3, 20, 1, 2, 1)
    y = model(A_, B_)
    print(y.size())

    print('Functional interface test passed')

    z = torch.mean(y)
    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')

    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda()

    y = Correlation(3, 3, 20, 1, 2, 1)(A_, B_)
    print(y.size())

    print('Module interface test passed')

    z = torch.mean(y)
    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')

def test_correlation_0():
    #model = correlation(0, 1, 0, 1, 1, 1)

    A = torch.Tensor([[1,2],[3,4]])
    B = torch.Tensor([[5,6],[7,8]])
    A = A.view((1,1,2,2))
    B = B.view((1,1,2,2))
    A = Variable(A)
    B = Variable(B)
    A_ = A.cuda()
    B_ = B.cuda()

    #y = model(A_, B_)
    #print(y) # should be 1x1x2x2 [[5,12],[21,32]]

    model2 = correlation(1, 1, 1, 1, 1, 1)
    y2 = model2(A_, B_)
    print(y2) # should be 1x9x2x2



def test_correlation1d_0():
    #model = correlation1d(0, 1, 0, 1, 1, 1)

    A = torch.Tensor([[1,2],[3,4]])
    B = torch.Tensor([[5,6],[7,8]])
    A = A.view((1,1,2,2))
    B = B.view((1,1,2,2))
    A = Variable(A)
    B = Variable(B)
    A_ = A.cuda()
    B_ = B.cuda()

    #y = model(A_, B_)
    #print(y) # should be 1x1x2x2 [[5,12],[21,32]]


    model2 = correlation1d(1, 1, 1, 1, 1, 1)
    y2 = model2(A_, B_)
    print(y2) # should be 1x3x2x2

    return


def test_correlation1d():
    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda()

    #import pdb; pdb.set_trace()
    model = correlation1d(20, 1, 20, 1, 1, 1)
    y = model(A_, B_)
    print(y.size())

    print('Functional interface test passed')

    z = torch.mean(y)
    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')

    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda()

    y = Correlation1d(20, 1, 20, 1, 1, 1)(A_, B_)
    print(y.size())

    print('Module interface test passed')

    z = torch.mean(y)
    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')


if __name__=='__main__':

    #test_correlation()

    #test_correlation1d()

    #test_correlation_0()

    test_correlation1d_0()
