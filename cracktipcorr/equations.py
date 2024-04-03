import sympy
from sympy.physics.units import millimeter, newton
from sympy.core.rules import Transform
from latex2sympy2 import latex2sympy
import numpy


def sympy_converter(formula, susb_dict=None, use_units=False, evalf=True, simplify=False, tolerance=0.01,
                    round_digits=None, latex_output=False):
    '''Convert a string to a SymPy expression.
    # SymPy functions:
    # evalf() is used to evaluate the expression numerically
    # subs() is used to substitute a value for a symbol
    # lambdify() is used to convert an expression into a function
    # nsimplify is used to simplify an expression

    Parameters
    ----------
    formula : str The formula to convert.
    susb_dict : dict, optional A dictionary with values to substitute.
    use_units : bool, optional If True, the resulting unit must be mm.
    evalf : bool, optional If True, the formula is evaluated numerically.
    simplify : bool, optional If True, the formula is simplified.
    tolerance : float, optional The tolerance for the simplify function.
    round_digits : int, optional The number of digits to round the formula to.
    latex_output : bool, optional If True, the formula is converted to LaTeX.


    Returns
    -------
    sympy expression The converted formula.
    '''

    # Replace variable names with SymPy symbols
    formula = formula.replace("a_(-3)", "a_3")
    formula = formula.replace("a_(-2)", "a_2")
    formula = formula.replace("a_(-1)", "a_1")
    formula = formula.replace("a_(0)", "a0")
    formula = formula.replace("a_(1)", "a1")
    formula = formula.replace("a_(2)", "a2")
    formula = formula.replace("a_(3)", "a3")
    formula = formula.replace("a_(4)", "a4")
    formula = formula.replace("a_(5)", "a5")
    formula = formula.replace("a_(6)", "a6")
    formula = formula.replace("a_(7)", "a7")
    formula = formula.replace("b_(-3)", "b_3")
    formula = formula.replace("b_(-2)", "b_2")
    formula = formula.replace("b_(-1)", "b_1")
    formula = formula.replace("b_(0)", "b0")
    formula = formula.replace("b_(1)", "b1")
    formula = formula.replace("b_(2)", "b2")
    formula = formula.replace("b_(3)", "b3")
    formula = formula.replace("b_(4)", "b4")
    formula = formula.replace("b_(5)", "b5")
    formula = formula.replace("b_(6)", "b6")
    formula = formula.replace("b_(7)", "b7")

    formula = formula.replace("X0", "a_1")
    formula = formula.replace("X1", "a0")
    formula = formula.replace("X2", "a1")
    formula = formula.replace("X3", "b_1")
    formula = formula.replace("X4", "b0")
    formula = formula.replace("X5", "b1")

    # Create SymPy symbols for variables
    ns = {}
    if use_units:
        ns["a_3"] = 1.0 * newton * millimeter ** (1 / 2)
        ns["a_2"] = 1.0 * newton * millimeter ** (0)
        ns["a_1"] = 1.0 * newton * millimeter ** (-1 / 2)
        ns["a0"] = 1.0 * newton * millimeter ** (-1)
        ns["a1"] = 1.0 * newton * millimeter ** (-3 / 2)
        ns["a2"] = 1.0 * newton * millimeter ** (-2)
        ns["a3"] = 1.0 * newton * millimeter ** (-5 / 2)
        ns["a4"] = 1.0 * newton * millimeter ** (-3)
        ns["a5"] = 1.0 * newton * millimeter ** (-7 / 2)
        ns["a6"] = 1.0 * newton * millimeter ** (-4)
        ns["a7"] = 1.0 * newton * millimeter ** (-9 / 2)
        ns["b_3"] = 1.0 * newton * millimeter ** (1 / 2)
        ns["b_2"] = 1.0 * newton * millimeter ** (0)
        ns["b_1"] = 1.0 * newton * millimeter ** (-1 / 2)
        ns["b0"] = 1.0 * newton * millimeter ** (-1)
        ns["b1"] = 1.0 * newton * millimeter ** (-3 / 2)
        ns["b2"] = 1.0 * newton * millimeter ** (-2)
        ns["b3"] = 1.0 * newton * millimeter ** (-5 / 2)
        ns["b4"] = 1.0 * newton * millimeter ** (-3)
        ns["b5"] = 1.0 * newton * millimeter ** (-7 / 2)
        ns["b6"] = 1.0 * newton * millimeter ** (-4)
        ns["b7"] = 1.0 * newton * millimeter ** (-9 / 2)


    else:
        ns["a_3"] = sympy.Symbol("A_{-3}")
        ns["a_2"] = sympy.Symbol("A_{-2}")
        ns["a_1"] = sympy.Symbol("A_{-1}")
        ns["a0"] = sympy.Symbol("A_0")
        ns["a1"] = sympy.Symbol("A_1")
        ns["a2"] = sympy.Symbol("A_2")
        ns["a3"] = sympy.Symbol("A_3")
        ns["a4"] = sympy.Symbol("A_4")
        ns["a5"] = sympy.Symbol("A_5")
        ns["a6"] = sympy.Symbol("A_6")
        ns["a7"] = sympy.Symbol("A_7")
        ns["b_3"] = sympy.Symbol("B_{-3}")
        ns["b_2"] = sympy.Symbol("B_{-2}")
        ns["b_1"] = sympy.Symbol("B_{-1}")
        ns["b0"] = sympy.Symbol("B_0")
        ns["b1"] = sympy.Symbol("B_1")
        ns["b2"] = sympy.Symbol("B_2")
        ns["b3"] = sympy.Symbol("B_3")
        ns["b4"] = sympy.Symbol("B_4")
        ns["b5"] = sympy.Symbol("B_5")
        ns["b6"] = sympy.Symbol("B_6")
        ns["b7"] = sympy.Symbol("B_7")

    ns.update({
        "add": sympy.Add,
        "sub": lambda x, y: x - y,
        "mul": sympy.Mul,
        "div": lambda x, y: x / y,
        "log": sympy.log,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "neg": lambda x: -1 * x,
        "inv": lambda x: 1 / x,
        "pow": sympy.Pow,
        "sqrt": sympy.sqrt
    })

    # Convert formula string to a SymPy expression
    sympy_formula = sympy.sympify(formula, locals=ns, evaluate=True)

    if evalf:
        sympy_formula = sympy.simplify(sympy_formula.evalf(), tolerance=0.1)

    if susb_dict is not None:
        sympy_formula = sympy_formula.subs(susb_dict)

    if simplify:
        sympy_formula = sympy.simplify(sympy_formula, tolerance=tolerance)

    if round_digits is not None:
        sympy_formula = sympy_formula.xreplace(
            Transform(lambda x: x.round(round_digits), lambda x: isinstance(x, sympy.Float)))

    if latex_output:
        sympy_latex = sympy.latex(sympy_formula)
        sympy_latex = sympy_latex.replace(r'\text', r'\mathrm')

    return sympy_latex, sympy_formula


def latex_to_sympy(latex_formula):
    '''Convert a LaTeX formula to a SymPy expression.

    Parameters
    ----------
    latex : str The LaTeX formula to convert.

    Returns
    -------
    sympy expression The converted formula.
    '''

    latex_formula = latex_formula.replace(r"\mathrm{mm}", "1")
    latex_formula = latex_formula.replace(r"\mathrm{N}", "1")
    sympy_formula = latex2sympy(latex_formula)
    a_3 = sympy.Symbol("A_{-3}")
    a_2 = sympy.Symbol("A_{-2}")
    a_1 = sympy.Symbol("A_{-1}")
    a0 = sympy.Symbol("A_0")
    a1 = sympy.Symbol("A_1")
    a2 = sympy.Symbol("A_2")
    a3 = sympy.Symbol("A_3")
    a4 = sympy.Symbol("A_4")
    a5 = sympy.Symbol("A_5")
    a6 = sympy.Symbol("A_6")
    a7 = sympy.Symbol("A_7")
    b_3 = sympy.Symbol("B_{-3}")
    b_2 = sympy.Symbol("B_{-2}")
    b_1 = sympy.Symbol("B_{-1}")
    b0 = sympy.Symbol("B_0")
    b1 = sympy.Symbol("B_1")
    b2 = sympy.Symbol("B_2")
    b3 = sympy.Symbol("B_3")
    b4 = sympy.Symbol("B_4")
    b5 = sympy.Symbol("B_5")
    b6 = sympy.Symbol("B_6")
    b7 = sympy.Symbol("B_7")
    lambdify_function = sympy.lambdify([a_3, a_2, a_1, a0, a1, a2, a3, a4, a5, a6, a7,
                                        b_3, b_2, b_1, b0, b1, b2, b3, b4, b5, b6, b7], sympy_formula, 'numpy')

    return sympy_formula, lambdify_function
