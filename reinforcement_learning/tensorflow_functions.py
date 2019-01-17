import tensorflow as tf
import numpy as np

def initialize_uninitialized(sess, worker_name):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print("\n\nInitializing uninitialized vars for worker : %s\n\n"%worker_name)
    print("\n\nnot_initialized_vars : %s\n\n"%[str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars)>0:
        sess.run(tf.variables_initializer(not_initialized_vars))

def build_histo_summary(values, bins):
    #Modified from https://www.programcreek.com/python/example/90429/tensorflow.HistogramProto
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)
    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))
    # Drop the start of the first bin
    bin_edges = bin_edges[1:]
    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    return(hist)

def update_target_graph(from_scope, to_scope):
    """
    Picked at https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder
