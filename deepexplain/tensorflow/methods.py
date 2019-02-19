from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from skimage.util import view_as_windows
import warnings
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad
from collections import OrderedDict
from .deep_shapley import eta_shap

SUPPORTED_ACTIVATIONS = [
    'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus'
]

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0
_MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

SESSION = None
FEED = None

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    #if op.type not in SUPPORTED_ACTIVATIONS:
    #    warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """
    def __init__(self, T, X, xs, session, keras_learning_phase=None):
        self.T = T
        self.X = X
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        print ('Model with multiple inputs: ', self.has_multiple_inputs)

    def session_run(self, T, xs):
        feed_dict = {}
        if self.has_multiple_inputs:
            if len(xs) != len(self.X):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   % (str(len(xs)), str(len(self.X))))
            for k, v in zip(self.X, xs):
                feed_dict[k] = v
        else:
            feed_dict[self.X] = xs
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.session)
        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        #print (feed_dict)
        FEED = feed_dict
        result = self.session.run(T, feed_dict)
        coord.request_stop()
        coord.join(threads)
        return result

    def _set_check_baseline(self):
        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros((1,) + xi.shape[1:]) for xi in self.xs]
            else:
                self.baseline = np.zeros((1,) + self.xs.shape[1:])

        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.xs):
                    if self.baseline[i].shape == self.xs[i].shape[1:]:
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    elif self.baseline[i].shape == self.xs[i].shape:
                        pass
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                           % (self.baseline[i].shape, self.xs[i].shape[1:]))
            else:
                if self.baseline.shape == self.xs.shape[1:]:
                    self.baseline = np.expand_dims(self.baseline, 0)
                elif self.baseline.shape == self.xs.shape:
                    pass
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                       % (self.baseline.shape, self.xs.shape[1:]))


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """
    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)

    def run(self):
        attributions = self.get_symbolic_attribution()
        results =  self.session_run(attributions, self.xs)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)

    @classmethod
    def matmul_grad_override(cls, op, grad):
        return original_grad(op, grad)

    @classmethod
    def convolution_grad_override(cls, op, grad):
        print ('conv: original grad')
        #print (op)
        #print (grad)
        #print (original_grad(op, grad))
        return original_grad(op, grad)

    @classmethod
    def maxpool_grad_override(cls, op, grad):
        print('maxpool: original gradient')
        #print(op)
        #print(grad)
        #print(original_grad(op, grad))
        return original_grad(op, grad)




class PerturbationBasedMethod(AttributionMethod):
    """
       Base class for perturbation-based attribution methods
       """
    def __init__(self, T, X, xs, session, keras_learning_phase):
        super(PerturbationBasedMethod, self).__init__(T, X, xs, session, keras_learning_phase)
        self.base_activation = None

    def _run_input(self, x):
        return self.session_run(self.T, x)

    def _run_original(self):
        return self._run_input(self.xs)

    def run(self):
        raise RuntimeError('Abstract: cannot run PerturbationBasedMethod')


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------
"""
Returns zero attributions. For testing only.
"""


class DummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)

"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""


class Saliency(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [tf.abs(g) for g in tf.gradients(self.T, self.X)]


"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


class GradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]


"""
Integrated Gradients
https://arxiv.org/pdf/1703.01365.pdf
"""


class IntegratedGradients(GradientBasedMethod):

    def __init__(self, T, X, xs, session, keras_learning_phase, steps=100, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, xs, session, keras_learning_phase)
        self.steps = steps
        self.baseline = baseline

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        attributions = self.get_symbolic_attribution()
        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [xs * alpha for xs in self.xs] if self.has_multiple_inputs else self.xs * alpha
            _attr = self.session_run(attributions, xs_mod)
            if gradient is None: gradient = _attr
            else: gradient = [g + a for g, a in zip(gradient, _attr)]

        results = [g * (x - b) / self.steps for g, x, b in zip(
            gradient,
            self.xs if self.has_multiple_inputs else [self.xs],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results


"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""


class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, xs, session, keras_learning_phase, epsilon=1e-4):
        super(EpsilonLRP, self).__init__(T, X, xs, session, keras_learning_phase)
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + eps *
                                tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))

"""
DeepLIFT
This reformulation only considers the "Rescale" rule
https://arxiv.org/abs/1704.02685
"""


class DeepLIFTRescale(GradientBasedMethod):

    _deeplift_ref = {}

    def __init__(self, T, X, xs, session, keras_learning_phase, baseline=None):
        super(DeepLIFTRescale, self).__init__(T, X, xs, session, keras_learning_phase)
        self.baseline = baseline

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                               original_grad(instant_grad.op, grad))

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        # Init references with a forward pass
        self._init_references()

        # Run the default run
        return super(DeepLIFTRescale, self).run()

    def _init_references(self):
        print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)

        YR = self.session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        print('DeepLIFT: references ready')
        sys.stdout.flush()


"""
Linear
"""


class Linear(GradientBasedMethod):
    def __init__(self, T, X, xs, session, keras_learning_phase):
        super(Linear, self).__init__(T, X, xs, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return grad




"""
DeepShapley
"""


class DeepShapley(GradientBasedMethod):

    _deepshap_ref = {}
    _deepshap_for = {}

    _override_op = {}
    _method = None
    # (key, value), where key is a string with the op name
    # and value if a dictionary {
    #  input: np.ndarray
    #  reference: np.ndarray
    #  weights: np.ndarray
    #  bias: np.ndarray
    # }

    def __init__(self, T, X, xs, session, keras_learning_phase, baseline=None):
        super(DeepShapley, self).__init__(T, X, xs, session, keras_learning_phase)
        self.baseline = baseline

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def convolution_grad_override(cls, op, grad):
        g1, g2 = original_grad(op, grad)
        if op.name not in cls._override_op:
            print("%s uses original gradient" % (op.name,))
            return g1, g2

        # print ('Conv2D override: ', op.name)

        players = cls._override_op[op.name]['input']
        reference = cls._override_op[op.name]['reference']
        kernel = cls._override_op[op.name]['weights']
        bias = cls._override_op[op.name]['bias']

        # Convert Conv2D into MatMul operation and proceed
        b = players.shape[0]
        ksizes = (1,) + kernel.shape[0:2] + (1,)
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        rates = op.get_attr('dilations')

        def extract_patches(x):
            return tf.extract_image_patches(
                x,
                ksizes=ksizes,
                strides=strides,
                rates=rates,
                padding=padding
            )

        def extract_patches_inverse(x, y):
            _x = tf.zeros_like(x)
            #print (_x)
            _y = extract_patches(_x)
            #print (_y)
            y = tf.check_numerics(
                y,
                'y contains nans',
            )
            grad = tf.gradients(_y, _x)[0]
            # Divide by grad, to "average" together the overlapping patches
            # otherwise they would simply sum up
            return tf.gradients(_y, _x, grad_ys=y)[0] / grad


        # Extract patches following same settings of convolution
        patches = extract_patches(players)
        reference = extract_patches(reference)

        # Reshape patches to have all elements involved in convolution together in the last dimension
        _players = tf.reshape(patches, (-1, np.prod(kernel.shape[:-1])))
        reference = tf.reshape(reference, (-1, np.prod(kernel.shape[:-1])))
        # Do the same for the kernel, except that we aggregate all kernel values in the first dimension
        weights = kernel.reshape(-1, kernel.shape[-1])

        _players = _players.eval(session=SESSION)
        reference = reference.eval(session=SESSION)

        # print ("Players", _players.shape)
        # print ("Reference", reference.shape)
        # print ("Kernel", weights.shape)
        # print ("Bias", bias.shape)

        grad = tf.reshape(grad, (-1, kernel.shape[-1]))
        # print ("Grad", grad.shape)


        eta = eta_shap(np.expand_dims(_players, -1) * weights,
                       baseline=np.expand_dims(np.repeat(reference, b, 0), -1) * weights,
                       bias=bias,
                       method='approx' if cls._method is None else 'revcancel',
                       )

        shap_weights = tf.expand_dims(weights, 0) * eta  # b, n ,m
        result = tf.reduce_sum(shap_weights * tf.expand_dims(grad, 1), -1)
        #result = tf.reduce_sum(tf.expand_dims(weights, 0) * eta, -1)


        # print ("Result,prereshape", result.shape)
        result = extract_patches_inverse(players, result)
        return result, g2

    @classmethod
    def maxpool_grad_override(cls, op, grad):
        print ("Max pool override")
        players = cls._override_op[op.name]['input']
        reference = cls._override_op[op.name]['reference']

        b, w, h, c = players.shape
        _, kw, kh, _ = op.get_attr('ksize')
        hw, hh = w // kw, h // kh
        pad = [[0, 0], [0, 0]]
        print ("About to run space_to_batch")

        x_ = tf.space_to_batch_nd(players, [kw, kh], pad)
        r_ = tf.space_to_batch_nd(reference, [kw, kh], pad)
        #print (x.shape)
        x = tf.reshape(x_, (kw * kh, -1))
        r = tf.reshape(r_, (kw * kh, -1))
        grad_flat = tf.reshape(grad, (-1, 1)) # all in batch
        x_np = tf.transpose(x, (1, 0)).eval(feed_dict=FEED, session=SESSION)
        r_np = tf.transpose(r, (1, 0)).eval(feed_dict=FEED, session=SESSION)
        print ("About to run eta_shap max")
        eta = eta_shap(np.expand_dims(x_np, -1),
                          baseline=np.expand_dims(np.repeat(r_np, b, 0), -1),
                          method='exact',
                          fun = lambda x: np.max(x, 1))
        print ("Finished to run eta_shap max")

        result = tf.reduce_sum(eta * tf.expand_dims(grad_flat, 1), -1)
        result = tf.transpose(result, (1, 0))
        result = tf.reshape(result, (kw * kh, b, -1))
        result = tf.reshape(result, (-1, hw, hh, c))
        result = tf.batch_to_space_nd(result, [kw, kh], pad)
        return result

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return grad

    @classmethod
    def matmul_grad_override(cls, op, grad):
        g1, g2 = original_grad(op, grad)
        print (cls._method)
        if op.name not in cls._override_op:
            print ("%s uses original gradient" % (op.name,))
            return g1, g2


        players = cls._override_op[op.name]['input']
        reference = cls._override_op[op.name]['reference']
        weights = cls._override_op[op.name]['weights']
        bias = cls._override_op[op.name]['bias']

        # print ('MatMul override: ', op.name)
        # print ("Players", players.shape)
        # print ("Weights", weights.shape)
        # print ("Reference", reference.shape)
        # print ("Original grad", g1.shape)

        # if 'dense_1' not in op.name:
        #     # Test: use normal deeplift here
        #     print ("Using DeepLIFT rescale for %s" % (op.name))
        #     input = tf.matmul(players, tf.transpose(weights)) + bias
        #     ref_input = tf.matmul(reference, tf.transpose(weights)) + bias
        #     output = tf.nn.relu(input)
        #     ref_output = tf.nn.relu(ref_input)
        #     delta_out = output - ref_output
        #     delta_in = input - ref_input
        #     instant_grad = tf.nn.relu(0.5 * (ref_input + input))
        #     return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
        #                     original_grad(instant_grad.op, grad)), g2




        eta = eta_shap(np.expand_dims(players, -1) * weights,
                          baseline=np.expand_dims(reference[0], -1) * weights,
                          bias=bias,
                          method='approx' if cls._method is None else 'revcancel',
                          )

        shap_weights = tf.expand_dims(weights, 0) * eta # b, n ,m
        g1shap = tf.reduce_sum(shap_weights * tf.expand_dims(grad, 1), -1)

        assert g1shap.get_shape().as_list()[1:] == g1.get_shape().as_list()[1:], \
            "Gradient got shape %s, while expecting %s" % (g1shap.get_shape().as_list(), g1.get_shape().as_list())
        return g1shap, g2

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        # Init references with a forward pass
        self._init_references()

        # Run the default run
        return super(DeepShapley, self).run()

    def _init_references(self):
        print ('Shapley: computing references...')
        sys.stdout.flush()
        self._override_op.clear()
        g = tf.get_default_graph()

        def run_baseline(output):
            return self.session_run([output], self.baseline)[0]

        def run_input(output):
            return self.session_run([output], self.xs)[0]

        for op in g.get_operations():
            # TODO: 'model' in op.name is not a good idea out of Keras! Maybe with SubGraphs?
            if len(op.inputs) > 0 and not 'gradients' in op.name and 'model' in op.name:

                # Handle non-linear activations
                if op.type in SUPPORTED_ACTIVATIONS:
                    nonlinear_op_name = op.name
                    main_op_name = None
                    self._override_op[nonlinear_op_name] = {
                        'input': None,
                        'reference': None,
                        'weights': None,
                        'bias': None
                    }

                    op = op.inputs[0].op
                    while op.inputs:
                        if op.type == 'BiasAdd':
                            self._override_op[nonlinear_op_name]['bias'] = run_baseline(op.inputs[1])
                        elif op.type == 'MatMul' or op.type == 'Conv2D':
                            self._override_op[nonlinear_op_name]['input'] = run_input(op.inputs[0])
                            self._override_op[nonlinear_op_name]['reference'] = run_baseline(op.inputs[0])
                            self._override_op[nonlinear_op_name]['weights'] = op.inputs[1].eval(session=SESSION)
                            main_op_name = op.name
                            break
                        else:
                            raise RuntimeError('Unexpected anchestor to non-linearity: ', op.name)
                        if len(op.inputs) == 0:
                            break
                        op = op.inputs[0].op
                    self._override_op[main_op_name] = self._override_op[nonlinear_op_name]
                    del self._override_op[nonlinear_op_name]

                elif op.type == 'MaxPool':
                    self._override_op[op.name] = {
                        'input': run_input(op.inputs[0]),
                        'reference': run_baseline(op.inputs[0]),
                        'weights': None,
                        'bias': None
                    }
        print ("Applying DeepShap on the following ops: ", self._override_op.keys())




    # def _init_references(self):
    #     print ('Shapley: computing references...')
    #     sys.stdout.flush()
    #     self._deepshap_ref.clear()
    #     ops = []
    #     tensors = []
    #     g = tf.get_default_graph()
    #     for op in g.get_operations():
    #         if len(op.inputs) > 0 and not 'gradients' in op.name and 'model' in op.name:
    #             print (op.type + ": " + op.name)
    #
    #             if op.type == 'MatMul':
    #                 ops.append(op.name + "_x")
    #                 tensors.append(op.inputs[0])
    #                 ops.append(op.name + "_w")
    #                 tensors.append(op.inputs[1])
    #             elif op.type == 'BiasAdd':
    #                 ops.append(op.name[:-7] + "MatMul_b")
    #                 tensors.append(op.inputs[1])
    #                 ops.append(op.name[:-7] + "convolution_b")
    #                 tensors.append(op.inputs[1])
    #             elif op.type == 'MaxPool':
    #                 ops.append(op.name + "_x")
    #                 tensors.append(op.inputs[0])
    #             elif op.type == 'Conv2D':
    #                 ops.append(op.name + "_x")
    #                 tensors.append(op.inputs[0])
    #                 ops.append(op.name + "_w")
    #                 tensors.append(op.inputs[1])
    #
    #     print (self.baseline.shape)
    #     #assert False
    #     YXS = self.session_run(tensors, self.xs)
    #     #YR = self.session_run(tensors, self.baseline)
    #     YR = self.session_run(tensors, self.xs)
    #     for (r, opName) in zip(YR, ops):
    #         self._deepshap_ref[opName] = np.mean(r, 0, keepdims=True)
    #     for (r, opName) in zip(YXS, ops):
    #         self._deepshap_for[opName] = r
    #     for k in self._deepshap_ref.keys():
    #         print (k, self._deepshap_ref[k].shape)
    #     print('Shapley: references ready')
    #     sys.stdout.flush()


"""
DeepLift (Reveal Cancel)
"""


class DeepLiftRC(DeepShapley):
    _method = "revcancel"
    def __init__(self, T, X, xs, session, keras_learning_phase, baseline=None):
        super(DeepLiftRC, self).__init__(T, X, xs, session, keras_learning_phase, baseline=baseline)


"""
Occlusion method
Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
This method performs a systematic perturbation of contiguous hyperpatches in the input,
replacing each patch with a user-defined value (by default 0).

window_shape : integer or tuple of length xs_ndim
Defines the shape of the elementary n-dimensional orthotope the rolling window view.
If an integer is given, the shape will be a hypercube of sidelength given by its value.

step : integer or tuple of length xs_ndim
Indicates step size at which extraction shall be performed.
If integer is given, then the step is uniform in all dimensions.
"""


class Occlusion(PerturbationBasedMethod):

    def __init__(self, T, X, xs, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, xs, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = xs[0].shape
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        print('Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self):
        self._run_original()

        input_shape = self.xs.shape[1:]
        batch_size = self.xs.shape[0]
        total_dim = np.asscalar(np.prod(input_shape))

        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(self.xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._run_original()

        # Start perturbation loop
        for i, p in enumerate(idx_patches):
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * self.xs
            delta = eval0 - self._run_input(masked_xs)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, self.xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                          'probably because window_shape and step do not allow to cover the all input.')
        return attribution


# -----------------------------------------------------------------------------
# END ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'elrp': (EpsilonLRP, 4),
    'deeplift': (DeepLIFTRescale, 5),
    'deeplift_rc': (DeepLiftRC, 6),
    'shapley': (DeepShapley, 7),
    'linear': (Linear, 8),
    'occlusion': (Occlusion, 9)
})


try:
    @ops.RegisterGradient("DeepExplainGrad")
    def deepexplain_grad(op, grad):
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        _GRAD_OVERRIDE_CHECKFLAG = 1
        if _ENABLED_METHOD_CLASS is not None \
                and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
            return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
        else:
            return original_grad(op, grad)
except:
    print ('WARNING: failed to register DeepExplainGrad. ALready there?')

try:
    @ops.RegisterGradient("MatMulDeepExplainGrad")
    def matmul_deepexplain_grad(op, grad):
        global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
        if _ENABLED_METHOD_CLASS is not None \
                and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
            return _ENABLED_METHOD_CLASS.matmul_grad_override(op, grad)
        else:
            return original_grad(op, grad)
except:
    print ('WARNING: failed to register MatMulDeepExplainGrad. ALready there?')

try:
    @ops.RegisterGradient("ConvolutionDeepExplainGrad")
    def convolution_deepexplain_grad(op, grad):
        global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
        if _ENABLED_METHOD_CLASS is not None \
                and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
            return _ENABLED_METHOD_CLASS.convolution_grad_override(op, grad)
        else:
            return original_grad(op, grad)
except:
    print ('WARNING: failed to register ConvolutionDeepExplainGrad. ALready there?')


try:
    @ops.RegisterGradient("MaxPoolDeepExplainGrad")
    def maxpool_deepexplain_grad(op, grad):
        global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
        if _ENABLED_METHOD_CLASS is not None \
                and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
            return _ENABLED_METHOD_CLASS.maxpool_grad_override(op, grad)
        else:
            return original_grad(op, grad)
except:
    print ('WARNING: failed to register MaxPoolDeepExplainGrad. ALready there?')


class DeepExplain(object):

    def __init__(self, graph=None, session=tf.get_default_session()):
        global SESSION
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')
        SESSION = session

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def explain(self, method, T, X, xs, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG

        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        print('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X, xs, self.session, self.keras_phase_placeholder, **kwargs)
        result = method.run()
        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

        self.keras_phase_placeholder = None
        return result

    def enable_override(self, method):
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        if method in attribution_methods:
            method_class, method_flag = attribution_methods[method]
            _ENABLED_METHOD_CLASS = method_class
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))

    def disable_override(self):
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        _ENABLED_METHOD_CLASS = None

    @staticmethod
    def get_override_map():
        map = dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)
        map['MatMul'] = 'MatMulDeepExplainGrad'
        map['Conv2D'] = 'ConvolutionDeepExplainGrad'
        map['MaxPool'] = 'MaxPoolDeepExplainGrad'
        return map

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.
        :return:
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]





