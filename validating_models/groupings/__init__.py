"""This package provides a bunch of grouping function. A grouping function is a function
mapping a :py:class:`validating_models.checker.Checker` and a list of indices :math:`I`, referring to instances 
in the dataset, to a tuple of a dictionary and a name for the grouping performed. 
The dictionary should map group names (strings) to lists of indices :math:`I_G \subseteq I` in the dataset.  
"""