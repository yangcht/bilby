import inspect
import types


def infer_parameters_from_function(func):
    """ Infers the arguments of a function
    (except the first arg which is assumed to be the dep. variable).

    Throws out `*args` and `**kwargs` type arguments

    Can deal with type hinting!

    Parameters
    ==========
    func: function or method
       The function or method for which the parameters should be inferred.

    Returns
    =======
    list: A list of strings with the parameters

    Raises
    ======
    ValueError
       If the object passed to the function is neither a function nor a method.

    Notes
    =====
    In order to handle methods the `type` of the function is checked, and
    if a method has been passed the first *two* arguments are removed rather than just the first one.
    This allows the reference to the instance (conventionally named `self`)
    to be removed.
    """
    if isinstance(func, types.MethodType):
        return infer_args_from_function_except_n_args(func=func, n=2)
    elif isinstance(func, types.FunctionType):
        return _infer_args_from_function_except_for_first_arg(func=func)
    else:
        raise ValueError("This doesn't look like a function.")


def infer_args_from_method(method):
    """ Infers all arguments of a method except for `self`

    Throws out `*args` and `**kwargs` type arguments.

    Can deal with type hinting!

    Returns
    =======
    list: A list of strings with the parameters
    """
    return infer_args_from_function_except_n_args(func=method, n=1)


def infer_args_from_function_except_n_args(func, n=1):
    """ Inspects a function to find its arguments, and ignoring the
    first n of these, returns a list of arguments from the function's
    signature.

    Parameters
    ==========
    func : function or method
       The function from which the arguments should be inferred.
    n : int
       The number of arguments which should be ignored, staring at the beginning.

    Returns
    =======
    parameters: list
       A list of parameters of the function, omitting the first `n`.

    Extended Summary
    ================
    This function is intended to allow the handling of named arguments
    in both functions and methods; this is important, since the first
    argument of an instance method will be the instance.

    See Also
    ========
    infer_args_from_method: Provides the arguments for a method
    infer_args_from_function: Provides the arguments for a function
    infer_args_from_function_except_first_arg: Provides all but first argument of a function or method.

    Examples
    ========

    .. code-block:: python

        >>> def hello(a, b, c, d):
        >>>     pass
        >>>
        >>> infer_args_from_function_except_n_args(hello, 2)
        ['c', 'd']

    """
    parameters = inspect.getfullargspec(func).args
    del parameters[:n]
    return parameters


def _infer_args_from_function_except_for_first_arg(func):
    return infer_args_from_function_except_n_args(func=func, n=1)


def get_dict_with_properties(obj):
    property_names = [p for p in dir(obj.__class__)
                      if isinstance(getattr(obj.__class__, p), property)]
    dict_with_properties = obj.__dict__.copy()
    for key in property_names:
        dict_with_properties[key] = getattr(obj, key)
    return dict_with_properties


def get_function_path(func):
    if hasattr(func, "__module__") and hasattr(func, "__name__"):
        return "{}.{}".format(func.__module__, func.__name__)
    else:
        return func


class PropertyAccessor(object):
    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. The properties of Interferometer are defined as instances
    of this class.

    This avoids lengthy code like

    .. code-block:: python

        @property
        def length(self):
            return self.geometry.length

        @length_setter
        def length(self, length)
            self.geometry.length = length

    in the Interferometer class
    """

    def __init__(self, container_instance_name, property_name):
        self.property_name = property_name
        self.container_instance_name = container_instance_name

    def __get__(self, instance, owner):
        return getattr(getattr(instance, self.container_instance_name), self.property_name)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.container_instance_name), self.property_name, value)


class _Repr:
    """
    Automatically generates a __repr__ string for a class.
    We use the inspect module to look at the __init__ function of the class
    and then look for attributes of the object with the same name as the
    arguments of the __init__ function. If they exist, we add them to the
    repr string.

    This takes into account nesting for class arguments that are themselves
    classes.

    Parameters
    ==========
    obj: object
        An instance of the class to generate the __repr__ string for.

    Returns
    =======
    repr: str
        The string representation of the object.
    """

    level = 0
    indent = "  "

    def __call__(self, obj):
        args = infer_args_from_method(obj.__init__)
        repr_string = f"{obj.__module__}.{obj.__class__.__name__}("
        if len(args) == 0:
            repr_string += ")"
        elif len(args) == 1:
            arg = args[0]
            value = _parse_single(arg, obj)
            repr_string += f"{arg}={value})"
        else:
            repr_string += "\n"
            self.level += 1
            for arg in args:
                try:
                    value = _parse_single(arg, obj)
                except AttributeError:
                    continue
                repr_string += f"{self.indent * self.level}{arg}={value},\n"
            self.level -= 1
            repr_string = repr_string.strip("\n").strip(",") + f"\n{self.indent * self.level})"
        return repr_string


def _parse_single(arg, obj):
    if hasattr(obj, arg):
        value = getattr(obj, arg)
    elif hasattr(obj, f"_{arg}"):
        value = getattr(obj, f"_{arg}")
    else:
        raise AttributeError
    if isinstance(value, (types.FunctionType, types.MethodType)):
        value = f"{value.__module__}.{value.__name__}"
    elif isinstance(value, str):
        value = f'"{value}"'
    return value


auto_repr = _Repr()
