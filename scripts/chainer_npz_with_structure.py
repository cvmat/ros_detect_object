
import chainer
import numpy
import six

# based on
# https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname#30042585
def import_class_from_string(path):
    from importlib import import_module
    module_path, _, class_name = path.rpartition('.')
    if module_path == '':
        klass = globals()[class_name]
    else:
        mod = import_module(module_path)
        klass = getattr(mod, class_name)
    return klass

def _convert_construction_info_to_string(obj):
    """
    Generate a string including information required to initialize an object
    in the same way of that of obj.

    Parameters
    ----------
    obj : object
        An object.
        It must have the attribute `_constructor_args` that represents
        the arguments of the constructor that has been used for initializing
        the object.

    Returns
    ----------
    encoded_construction_info : string
        It represents the module, class name and constructor arguments of obj.
    """
    if not hasattr(obj, '_constructor_args'):
        raise AttributeError('obj has no attribute _constructor_args.')
    import StringIO
    output = StringIO.StringIO()
    info = {}
    info['_module_name'] = obj.__class__.__module__
    info['_class_name'] = obj.__class__.__name__
    encoded_constructor_args = {}
    for k, v in obj._constructor_args.items():
        if isinstance(v, chainer.Link):
            encoded_constructor_args[k] \
                = _convert_construction_info_to_string(v)
        elif isinstance(v, six.string_types):
            encoded_constructor_args[k] = 'STR' + v
        else:
            encoded_constructor_args[k] = v
    info['_encoded_constructor_args'] = encoded_constructor_args
    numpy.save(output, arr=info)
    encoded_construction_info = 'OBJ' + output.getvalue()
    output.close()
    return encoded_construction_info

def _restore_obj_from_construction_info_string(str, class_name_replacement_list = None):
    """
    Restore an object from a construction info string.

    Parameters
    ----------
    str : string
        It represents the module, class name and constructor arguments
        of an object.

    class_name_replacement_list : list
        A list of a tuple of pattern and replacement.
        If a class name with a module matches a pattern in the list,
        the string is replaced by the replacement.
        The patterns are compared in the order of the list and only
        the first replacement will be applied.

    Returns
    -------
    obj : object
        It is initialized according to the construction info.
    """
    if str[0:3] == 'STR':
        obj = str[3:]
    elif str[0:3] == 'OBJ':
        import StringIO
        inp = StringIO.StringIO(str[3:])
        info = numpy.load(inp).item()
        inp.close()
        constructor_args = {}
        # Decode the encoded constructor arguments.
        for k, v in info['_encoded_constructor_args'].items():
            if isinstance(v, six.string_types):
                constructor_args[k] \
                    = _restore_obj_from_construction_info_string(
                        v, class_name_replacement_list)
            else:
                constructor_args[k] = v
        module_name = info['_module_name']
        class_name = info['_class_name']
        full_class_name = module_name + '.' + class_name
        original_full_class_name = full_class_name
        if not class_name_replacement_list is None:
            import re
            for pat, repl in class_name_replacement_list:
                if re.match(pat, full_class_name):
                    full_class_name = re.sub(pat, repl, full_class_name)
                    break
        class_object = import_class_from_string(full_class_name)
        obj = make_serializable_object(class_object, constructor_args)
    else:
        raise ValueError()
    return obj

def make_serializable_object(class_object, constructor_args, template_args=None):
    """
    Generate an object and attach the argument of the constructor to
    to the object.

    Parameters
    ----------
    class_object : class object
    constructor_args : dict
        Arguments used for generating an object of the given class.
    template_args : dict
        Arguments used for generating a template of an object of the
        given class in advance of loading trained weights.

    Returns
    -------
    obj : object
        It has an additional attribute `_constructor_args` that refers
        to template_args or constructor_args.
    """
    obj = class_object(**constructor_args)
    obj._constructor_args = template_args or constructor_args
    return obj

def save_npz_with_structure(file, obj, compression=True):
    """
    Save an object to the file in NPZ format.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
    obj : an object
        An object to be saved to the file. It must have an attribute
        _constructor_args that is the argument of the constructor for
        generating the object.
    compression : bool, optional
        If ``True``, the object is saved in compressed .npz format.
    """
    s = chainer.serializers.npz.DictionarySerializer()
    s.save(obj)
    if not hasattr(obj, '_constructor_args'):
        raise AttributeError('obj has no attribute _constructor_args.')
    #
    s.target['_encoded_construction_info'] \
        = _convert_construction_info_to_string(obj)
    #
    if compression:
        numpy.savez_compressed(file, **s.target)
    else:
        numpy.savez(file, **s.target)


def load_npz_with_structure(file, path = '', strict = True, class_name_replacement_list = None):
    """
    Load a model from a ``.npz`` file.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
    path : string, optional
        The base path that the deserialization starts from.
        This is sent to ``chainer.serializers.NpzDeserializer`` as is.
    strict : bool, optional
        If ``True``, the deserializer raises an error when an
        expected value is not found in the given NPZ file. Otherwise,
        it ignores the value and skip deserialization.
        This is sent to ``chainer.serializers.NpzDeserializer`` as is.
    class_name_replacement_list : list
        A list of a tuple of pattern and replacement.
        If the class name written in the file matches a pattern in the list,
        the class name is replaced by the replacement.
        The patterns are compared in the order of the list and only
        the first replacement will be applied.

    Returns
    -------
    obj : object
        The loaded object.
    """
    with numpy.load(file) as f:
        d = chainer.serializers.npz.NpzDeserializer(f,path=path,strict=strict)
        if not '_encoded_construction_info' in f:
            raise AttributeError(
                'The given file (%s) has no attribute '
                '_encoded_construction_info.' % (file,))
        encoded_initial_object = f['_encoded_construction_info'].item()
        obj = _restore_obj_from_construction_info_string(
            encoded_initial_object, class_name_replacement_list)
        d.load(obj)
    return obj


def modify_constructor_args_embedded_in_npz(input_file, output_file, new_constructor_args, path = '', strict = True, class_name_replacement_list = None):
    """
    Modify constructor args of a model from a ``.npz`` file and save it to the other file.

    Parameters
    ----------
    input_file : file-like object, string, or pathlib.Path
    output_file : file-like object, string, or pathlib.Path
    new_constructor_args : dict
        Arguments used for generating an object of the given class.
        It will be used when loading a model from the file specified
        by ``output_file``.
    path : string, optional
        The base path that the deserialization starts from.
        This is sent to ``chainer.serializers.NpzDeserializer`` as is.
    strict : bool, optional
        If ``True``, the deserializer raises an error when an
        expected value is not found in the given NPZ file. Otherwise,
        it ignores the value and skip deserialization.
        This is sent to ``chainer.serializers.NpzDeserializer`` as is.
    class_name_replacement_list : list
        A list of a tuple of pattern and replacement.
        If the class name written in the file matches a pattern in the list,
        the class name is replaced by the replacement.
        The patterns are compared in the order of the list and only
        the first replacement will be applied.

    Returns
    -------
    obj : object
        The loaded object.
    """
    obj = load_npz_with_structure(input_file)
    obj._constructor_args = new_constructor_args
    save_npz_with_structure(output_file, obj)
    return obj

# Example usage
if __name__ == '__main__':
    import chainer
    import chainer.links as L
    import chainer.functions as F
    import numpy as np
    #
    # Test for `chainer.links.Linear`.
    print('\nTest of saving/loading a `chainer.links.Linear` object.')
    filename = 'test.npz'
    np.random.seed(0)
    #
    # Prepare a serializable object.
    # Arguments of a constructor must be given as a dict.
    obj1 = make_serializable_object(L.Linear, {'in_size': 3, 'out_size': 3})
    obj1.W.data = np.random.uniform(-10, 10, (3, 3))
    #
    # Save the object.
    save_npz_with_structure(filename, obj1)
    #
    # It can be loaded without a template object.
    obj2 = load_npz_with_structure(filename)
    #
    # The object can be loaded by `chainer.serializers.load_npz()`.
    obj3 = L.Linear(**(obj1._constructor_args))
    chainer.serializers.load_npz(filename, obj3)
    #
    # The original model `obj1` and the loaded models `obj2` and `obj3`
    # are equivalent.
    np.random.seed(0)
    inps = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)
    oups1 = obj1(inps).data
    oups2 = obj2(inps).data
    oups3 = obj3(inps).data
    print('`obj1` is an original `Linear` object.')
    print('The object is saved to `%s` by `save_npz_with_structure()`.'
          % (filename,))
    print('`obj2` is loaded from `%s` by `load_npz_with_structure()`.'
          % (filename, ))
    print('`obj3` is loaded from `%s` by `chainer.serializers.load_npz()`.'
          % (filename, ))
    print('`oups1`, `oups2` and `oups3` are results calculated '
          'for a common input by `obj1`, `obj2` and `obj3`, respectively.')
    print('norm of (oups1 - oups2) = %e' % np.linalg.norm(oups1 - oups2))
    print('norm of (oups1 - oups3) = %e' % np.linalg.norm(oups1 - oups3))
    #
    #
    #
    # The functions can be applied to more complex objects.
    # The below is an example of a model that is initialized with other
    # models.
    print('\nTest of saving/loading a `chainer.links.Classifier` object.')
    #
    def lossfun(x, t):
        return F.softmax_cross_entropy(x, t)
    #
    np.random.seed(0)
    #
    # You should initialize a `chainer.Link` object by
    # `make_serializable_object` if it is referred in arguments
    # of the constructor.
    predictor = make_serializable_object(
        L.Linear, {'in_size': 5, 'out_size': 4},
        template_args = {
            'in_size': 5, 'out_size': 4,
            'initialW': 100, 'initial_bias': -1000,
        }
    )
    model1 = make_serializable_object(L.Classifier, {
        'predictor': predictor, 'lossfun': lossfun })
    #
    # Save the model.
    save_npz_with_structure('test_model.npz', model1)
    #
    # Load the model without a template object.
    model2 = load_npz_with_structure('test_model.npz')
    #
    # It can also be loaded by `chainer.serializers.load_npz()`.
    model3 = L.Classifier(L.Linear(5, 4), lossfun)
    chainer.serializers.load_npz('test_model.npz', model3)
    #
    #
    # Comparison:
    np.random.seed(0)
    inps = np.random.uniform(-1, 1, (3, 5)).astype(np.float32)
    labels = np.random.choice(range(0, 3), (3,)).astype(np.int32)
    oups1 = model1(inps, labels).data
    oups2 = model2(inps, labels).data
    oups3 = model3(inps, labels).data
    print('`model1` is an original `chainer.links.Classifier` object.')
    print('The object is saved to `%s` by `save_npz_with_structure()`.'
          % (filename,))
    print('`model2` is loaded from `%s` by `load_npz_with_structure()`.'
          % (filename, ))
    print('`model3` is loaded from `%s` by `chainer.serializers.load_npz()`.'
          % (filename, ))
    print('`oups1`, `oups2` and `oups3` are results calculated '
          'for a common input by `model1`, `model2` and `model3`, '
          'respectively.')
    print('norm of (oups1 - oups2) = %e' % np.linalg.norm(oups1 - oups2))
    print('norm of (oups1 - oups3) = %e' % np.linalg.norm(oups1 - oups3))

