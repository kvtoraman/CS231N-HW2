from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = np.zeros((x.shape[0],b.size))
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #print("x",x.shape)
    #print("b",b.shape)
    #print("out",out.shape)
    
    #for i in range(x.shape[0]):
    #    out[i] = w.T.dot(x[i].T) + b
    out = x.reshape(x.shape[0],np.prod(x[0].shape)).dot(w) + b.T   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: ???
      
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    #print("x.shape",x.shape)
    N,M = dout.shape
    D,M = w.shape
    #print("N,M,D",N,M,D)
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0],np.prod(x[0].shape)).T,dout)
    db = np.dot(np.ones(N).T,dout)
    
    #print("db shape",db.shape,db)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #res_x = x
    #x = x.reshape(-1)
    #print("x:",x.shape)
    #print("r:",res_x.shape)
    #out = np.zeros(len(x))
    # for i in range(len(x)):
    #     out[i] = max(0,x[i])
    out = np.maximum(0,x)
    
    #x = res_x
    #out = out.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    #print("x.shape",x.shape)
    #print(dout.shape)
#    res_x = x
#    res_dout = dout
#    dout = dout.reshape(-1)
#    x = x.reshape(-1)
    
#    dx = np.zeros(len(x))
#    for i in range(len(x)):
#        if x[i] > 0:
#            dx[i] = dout[i]
            
    dx = dout * (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


from numpy import linalg as LA
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        
        mean_v = np.mean(x, axis = 0)
        var_v = np.var(x, axis = 0)
        org_x = x
        # for  i in range(N):
        #     for j in range(D):
        #         #print("x[i] shape",x[i].shape)
        #         #print("mean shape",mean_v.shape)
        #         x[i][j] = (x[i][j] - mean_v[j]) / np.sqrt(var_v[j] + eps) #hope var**2 works elemwise
        x_norm = (x - mean_v) / np.sqrt(var_v + eps)
        x = x_norm
        out = np.zeros((N, D))
        # for  i in range(N):
        #     out[i] = np.multiply(gamma.T, x[i]) + beta.T
        out = np.multiply(gamma.T, x) + beta.T

        running_mean = momentum * running_mean + (1 - momentum) * mean_v
        running_var = momentum * running_var + (1 - momentum) * var_v
        cache = (org_x, x, gamma, beta, mean_v, var_v, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x = (x - running_mean) / np.sqrt(running_var + eps) #hope var**2 works elemwise
        
        out = np.zeros((N, D))
        # for  i in range(N):
        #     out[i] = np.multiply(gamma.T, x[i]) + beta.T
        out = np.multiply(gamma.T, x) + beta.T
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    N,D = dout.shape
    org_x, x, gamma, beta, mu, var, eps = cache
    
    dx = np.zeros((N, D))
    dx_norm = dout * gamma #elementwise
    std_inv = 1.0/np.sqrt(var + eps)
    
    dvar = np.sum(dx_norm * (org_x - mu), axis = 0) * -0.5 * std_inv ** 3
    dmu = np.sum(dx_norm* -std_inv, axis = 0) + dvar * np.mean(-2 * (org_x-mu),axis = 0)
    
    dx = (dx_norm * std_inv) + (dvar *2 * (org_x - mu) / N) + (dmu / N)
    
    
    dgamma = np.sum(dout * x, axis = 0)
#     for j in range(D):
#         for k in range(N):
#             dgamma[j] += dout[k][j] * x[k][j]
    
    dbeta = np.sum(dout, axis = 0)
#     dbeta = np.zeros(D)
#     for j in range(D):
#         for k in range(N):
#             dbeta[j] += dout[k][j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) > p) / p
        out = x*mask        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = (np.random.rand(*x.shape) > p) / p
        out = x * mask
        mask = None
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        ####################w###################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    # print("F, C, HH, WW",F, C, HH, WW)
    # print("N, C, H, W", N, C, H, W)
    # print("stride pad:",stride,pad)
    
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    
    out = np.zeros((N, F, H_prime, W_prime))
    incr = 0
    
    x_pad = np.zeros((N, C, H + pad*2, W + pad*2))
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    
    #print("x shape",x.shape)
    #print("x_pad",x_pad.shape)
    
    for i in range(N):#each image
        for f in range(F): #each filter
            for h in range(0, H_prime):
                for wi in range(0, W_prime):
                    for c in range(C):#each channel
                      #  for hh in range(HH):
                       #     for ww in range(WW):
                        #incr = w[f][c][hh][ww] * x_pad[i][c][h*stride - (hh - HH//2) + pad][wi*stride - (ww - WW//2) + pad]
                        small_x = x_pad[i,c,h*stride : h*stride + HH, wi*stride:wi*stride + WW]
                        #print("HH,WW,smallx shape",HH,WW,small_x.shape)
                        #print("x_pad",x_pad.shape)
    
                        #print("0,4 ? ",wi*stride , wi*stride + WW)
                        incr = np.sum(small_x * w[f][c])
                        #if incr > 0:
                        #    print(i,f,h//stride,wi//stride)
                        out[i][f][h][wi] += incr
                        
                    out[i][f][h][wi] += b[f]
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    print(dout.shape)
    print(x.shape,w.shape,b.shape)
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    N,F,H_prime,W_prime = dout.shape
    F, C, HH, WW = w.shape 
    N, C, H, W = x.shape
    
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    dx = np.zeros(x_pad.shape)
    
    for i in range(N):
        for f in range(F):
            for h in range(H_prime):
                for wi in range(W_prime):
                    
                    #for c in range(C):
                    small_x = x_pad[i,:,h*stride : h*stride + HH, wi*stride:wi*stride + WW]

                    dw[f] += dout[i][f][h][wi]*small_x
                    incr_dx = w[f] * dout[i][f][h][wi]
                    #print(incr_dx.shape,dx[i,c,h*stride:h*stride + HH,wi*stride:wi*stride + WW].shape)
                    dx[i,:,h*stride:h*stride + HH,wi*stride:wi*stride + WW] += incr_dx
                        
                    db[f] += dout[i][f][h][wi]
    #remove pads
    dx = dx[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height,pool_width,stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    H_prime = int(1 + (H - pool_height)//stride)
    W_prime = int(1 + (W - pool_width)//stride)
    out = np.zeros((N, C, H_prime, W_prime))
    index_cache = np.zeros(list(out.shape) + [2],dtype="int32")
    
    for i in range(N):
        for c in range(C):
            for h in range(H_prime):
                for wi in range(W_prime):
                    #out[i][c][h][wi] = np.amax([0, 0, 1])
                    index = np.argmax(x[i,c,h*stride : h*stride + pool_height,wi*stride : wi*stride +pool_width])
                    #print(index)
                    #print(type(index))
                    index_cache[i][c][h][wi] = np.array((index//pool_width,index%pool_width),dtype="int32")
                    out[i][c][h][wi] = x[i, c,  h*stride + index//pool_width, wi*stride +index%pool_width]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, index_cache)
    return out, cache




def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param,index_cache) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param,index_cache = cache
    pool_height,pool_width,stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    N, C, H_prime, W_prime = dout.shape
    
    N, C, H, W = x.shape
    dx = np.zeros_like(x)
    for i in range(N):
        for c in range(C):
            for h in range(H_prime):
                for wi in range(W_prime):
                    
                    h_os, w_os = index_cache[i][c][h][wi] 
                    #print(h_os,w_os)
                    dx[i][c][h*stride + h_os][wi*stride + w_os] = dout[i][c][h][wi]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #size of mean & var should be C
    N, C, H, W = x.shape
    #print(x.shape)
    new_x = x.swapaxes(0,1).reshape(C,-1)
    #print( new_x.shape)
    out,cache = batchnorm_forward(new_x.T, gamma, beta, bn_param)
    out = out.T.reshape(C,N,H,W).swapaxes(0,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.swapaxes(0,1).reshape(C,-1)
    dx,dgamma,dbeta = batchnorm_backward(dout.T,cache)
    dx = dx.T.reshape((C, N, H, W)).swapaxes(0,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
