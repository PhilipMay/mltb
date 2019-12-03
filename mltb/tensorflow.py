import tensorflow as tf

def set_gpu_mem_growth():
    """
    Only grow the memory usage as is needed by the process.

    This code is for tensorflow 2. It does not work with tensorflow 1.

    See Also
    --------
    * `Limiting GPU memory growth <https://www.tensorflow.org/beta/guide/using_gpu#limiting_gpu_memory_growth>`_
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
