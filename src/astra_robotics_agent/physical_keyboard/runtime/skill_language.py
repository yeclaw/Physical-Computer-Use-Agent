"""Interpret an explicit Python AST subset without executing Python code objects.

This is a bounded skill language, not arbitrary Python or an OS security sandbox.
No imports, reflection, classes, comprehensions, decorators, or object methods
other than the four base robot capabilities, explicitly enabled optional
capabilities and listed math functions are available.
Only ``except ToolError [as name]`` is supported; the binding can be rendered by
str(), and has no attributes. Unhandled skill errors cannot be suppressed by a
finally return. A host SkillError or exhausted budget latches execution closed.

Limits apply to each JSON value as well as cumulative allocations: item and
character allocation caps are respectively max_container_items*32 and
max_string_chars*32. JSON depth is at most 32 and expression/AST depth at most
64. These conservative caps can reject programs that ordinary Python accepts.
Callbacks and wall-clock/process deadlines remain the trusted host's concern.
There is no default robot-method-count cap. An explicit finite max_tool_calls
remains available for historical fixtures; normal execution uses None.
"""

import ast
import json
import math
from dataclasses import dataclass


class SkillError(ValueError):
    pass


class ValidationError(SkillError):
    pass


class BudgetExceeded(SkillError):
    pass


class ToolError(SkillError):
    """A recoverable tool rejection explicitly selected by the trusted host."""


@dataclass(frozen=True)
class Limits:
    max_steps: int = 20000
    max_iterations: int = 2048
    max_call_depth: int = 32
    max_tool_calls: int | None = None
    max_container_items: int = 1024
    max_string_chars: int = 4096
    max_int_bits: int = 128
    max_ast_nodes: int = 4096
    max_source_bytes: int = 65536

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == 'max_tool_calls' and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValidationError('Limits must be positive integers: ' + name)
        # These host-side caps prevent callers accidentally disabling all guards.
        if (self.max_call_depth > 64 or self.max_ast_nodes > 32768
                or self.max_source_bytes > 1024 * 1024 or self.max_int_bits > 1024
                or self.max_container_items > 65536 or self.max_string_chars > 65536):
            raise ValidationError('Limits exceed interpreter implementation caps')


_TOOLS = {'observe': (), 'move_tcp': ('xyz_m', 'quaternion_xyzw'),
          'wait_bounded': ('seconds',), 'stop': ()}
_DEMO_TOOLS = {'trajectory_catalog': (),
               'inspect_trajectory': ('trajectory_id', 'start_index', 'count'),
               'play_trajectory': ('trajectory_id', 'options'),
               'play_sequence': ('keys', 'options'),
               'preview_sequence': ('keys', 'options')}


def tool_specs(*, demo_tools=False):
    """Optional capabilities never become requirements of the base language."""
    if type(demo_tools) is not bool:
        raise ValidationError('demo_tools must be an explicit boolean')
    return dict(_TOOLS, **(_DEMO_TOOLS if demo_tools else {}))

_MATH = {name: getattr(math, name) for name in (
    'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'hypot',
    'fabs', 'floor', 'ceil', 'radians', 'degrees', 'isfinite')}
_BUILTINS = {'len', 'abs', 'min', 'max', 'sum', 'round', 'float', 'int', 'bool',
             'str', 'list', 'dict', 'range'}
_RESERVED = _BUILTINS | {'math', 'ToolError'}
_FORBIDDEN = {'open', 'eval', 'exec', 'compile', 'globals', 'locals', 'getattr',
              'setattr', 'delattr', 'hasattr', 'vars', 'type', 'object', 'super',
              'dir', 'input', 'print', 'help', 'exit', 'quit', 'memoryview',
              'bytes', 'bytearray', 'classmethod', 'staticmethod', 'property'}
_ALLOWED = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
    ast.Expr, ast.If, ast.For, ast.While, ast.Return, ast.Break, ast.Continue,
    ast.Try, ast.ExceptHandler, ast.Pass, ast.Name, ast.Constant, ast.List,
    ast.Dict, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Attribute, ast.Load,
    ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
}


def _public(name):
    if (not isinstance(name, str) or not name.isidentifier() or name.startswith('_')
            or name in _FORBIDDEN):
        raise ValidationError('Only public identifiers are supported')


def _constant_expression(node):
    return isinstance(node, (ast.Constant, ast.List, ast.Dict, ast.BinOp, ast.UnaryOp)) and all(
        isinstance(n, (ast.Constant, ast.List, ast.Dict, ast.BinOp, ast.UnaryOp,
                       ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                       ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not)) for n in ast.walk(node))


def _target(node):
    if isinstance(node, ast.Name):
        _public(node.id)
        if node.id in _RESERVED:
            raise ValidationError('Cannot assign reserved name: ' + node.id)
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            raise ValidationError('Slice assignment is not supported')
    else:
        raise ValidationError('Assignment target must be a name or subscript')


def _validate(tree, limits, callable_names, *, demo_tools=False):
    allowed_tools = tool_specs(demo_tools=demo_tools)
    stack = [(tree, 0, False, 0)]
    while stack:
        node, depth, in_function, loops = stack.pop()
        if depth > 64:
            raise ValidationError('AST nesting exceeds 64 levels')
        if type(node) not in _ALLOWED:
            raise ValidationError('Unsupported syntax: ' + type(node).__name__)
        if isinstance(node, ast.Name):
            _public(node.id)
        elif isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (str, int, float, bool, type(None)):
                raise ValidationError('Unsupported literal')
            if type(value) is int and value.bit_length() > limits.max_int_bits:
                raise ValidationError('Integer literal exceeds bit budget')
            if type(value) is float and not math.isfinite(value):
                raise ValidationError('Non-finite literal')
            if isinstance(value, str):
                try:
                    value.encode('utf-8')
                except UnicodeError as exc:
                    raise ValidationError('Invalid Unicode literal') from exc
                if len(value) > limits.max_string_chars:
                    raise ValidationError('String literal exceeds character budget')
        elif isinstance(node, ast.FunctionDef):
            if in_function:
                raise ValidationError('Nested function definitions are not supported')
            _public(node.name)
            if node.name in _RESERVED or node.decorator_list or node.returns or getattr(node, 'type_params', []):
                raise ValidationError('Unsupported function declaration')
            args = node.args
            if args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults:
                raise ValidationError('Only plain positional parameters are supported')
            names = [a.arg for a in args.args]
            if len(names) != len(set(names)):
                raise ValidationError('Duplicate parameter')
            for argument in args.args:
                _public(argument.arg)
                if argument.annotation is not None or argument.arg in _RESERVED:
                    raise ValidationError('Unsupported parameter declaration')
        elif isinstance(node, ast.Attribute):
            if (not isinstance(node.value, ast.Name)
                    or (node.value.id == 'robot' and node.attr not in allowed_tools)
                    or (node.value.id == 'math' and node.attr not in set(_MATH) | {'pi'})
                    or node.value.id not in {'robot', 'math'}):
                raise ValidationError('Object attributes are not available')
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValidationError('Indirect call expression is not supported')
            if any(k.arg is None for k in node.keywords):
                raise ValidationError('Argument unpacking is not supported')
            names = [k.arg for k in node.keywords]
            if len(names) != len(set(names)):
                raise ValidationError('Duplicate keyword argument')
            if isinstance(node.func, ast.Name) and node.func.id not in callable_names:
                raise ValidationError('Call must name a declared helper or permitted builtin')
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _target(target)
        elif isinstance(node, ast.AugAssign):
            _target(node.target)
        elif isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise ValidationError('Dictionary unpacking is not supported')
        elif isinstance(node, (ast.List, ast.Dict)) and len(node.elts if isinstance(node, ast.List) else node.keys) > limits.max_container_items:
            raise ValidationError('Container literal exceeds item budget')
        elif isinstance(node, ast.Return) and not in_function:
            raise ValidationError('Return outside function')
        elif isinstance(node, (ast.Break, ast.Continue)) and not loops:
            raise ValidationError('Loop control outside loop')
        elif isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise ValidationError('For target must be a name')
            _target(node.target)
        elif isinstance(node, ast.ExceptHandler):
            if not isinstance(node.type, ast.Name) or node.type.id != 'ToolError':
                raise ValidationError('Only except ToolError is supported')
            if node.name:
                _public(node.name)
                if node.name in _RESERVED:
                    raise ValidationError('Reserved exception binding')
        child_function = in_function or isinstance(node, ast.FunctionDef)
        child_loops = loops + isinstance(node, (ast.For, ast.While))
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1, child_function, child_loops))


@dataclass
class _Function:
    node: ast.FunctionDef


@dataclass(frozen=True)
class _Callable:
    kind: str
    name: str


class _Robot:
    pass


class _Math:
    pass


class _Flow(BaseException):
    def __init__(self, kind, value=None):
        self.kind, self.value = kind, value


class _Interpreter:
    def __init__(self, tools, limits, *, demo_tools=False):
        self.tools, self.limits = tools, limits
        self.tool_specs = tool_specs(demo_tools=demo_tools)
        self.steps = self.iterations = self.calls = self.tool_calls = self.expression_depth = 0
        self.allocated_items = self.allocated_chars = 0
        self.fatal = None
        self.globals = {name: _Callable('builtin', name) for name in _BUILTINS}
        self.globals['math'] = _Math()

    def fail_budget(self, message):
        self.fatal = BudgetExceeded(message)
        raise self.fatal

    def tick(self):
        if self.fatal is not None:
            raise self.fatal
        self.steps += 1
        if self.steps > self.limits.max_steps:
            self.fail_budget('Statement/expression step budget exhausted')

    def allocate(self, items=0, chars=0):
        self.allocated_items += items
        self.allocated_chars += chars
        if (self.allocated_items > self.limits.max_container_items * 32
                or self.allocated_chars > self.limits.max_string_chars * 32):
            self.fail_budget('Cumulative allocation budget exhausted')

    def check(self, value):
        stack, items, chars = [(value, 0, frozenset())], 0, 0
        while stack:
            value, depth, ancestors = stack.pop()
            if depth > 32:
                self.fail_budget('JSON nesting exceeds 32 levels')
            kind = type(value)
            if kind is int:
                if value.bit_length() > self.limits.max_int_bits:
                    self.fail_budget('Integer bit budget exhausted')
            elif kind is float:
                if not math.isfinite(value):
                    raise SkillError('Non-finite value')
            elif kind is str:
                if len(value) > self.limits.max_string_chars:
                    self.fail_budget('String character budget exhausted')
                try:
                    value.encode('utf-8')
                except UnicodeError as exc:
                    raise SkillError('Invalid Unicode data') from exc
                chars += len(value)
            elif kind in (list, dict):
                if id(value) in ancestors:
                    raise SkillError('Cyclic values are not JSON data')
                items += len(value)
                if items > self.limits.max_container_items:
                    self.fail_budget('Container item budget exhausted')
                ancestors = ancestors | {id(value)}
                if kind is dict:
                    if any(type(key) is not str for key in value):
                        raise SkillError('Dictionary keys must be strings')
                    children = [element for pair in value.items() for element in pair]
                else:
                    children = value
                stack.extend((child, depth + 1, ancestors) for child in children)
            elif kind not in (bool, type(None)):
                raise SkillError('Value is not pure JSON data')
        return items, chars

    def clone(self, value):
        items, chars = self.check(value)
        self.allocate(items, chars)
        def copied(v):
            if type(v) is list:
                return [copied(child) for child in v]
            if type(v) is dict:
                return {key: copied(child) for key, child in v.items()}
            return v
        return copied(value)

    @staticmethod
    def lookup(name, local, globals_):
        if name in local:
            return local[name]
        if name in globals_:
            return globals_[name]
        raise SkillError('Unknown name: ' + name)

    def data(self, value):
        self.check(value)
        return value

    def number(self, value):
        if type(value) not in (int, float):
            raise SkillError('Numeric operand required, not bool')
        self.check(value)
        return value

    def binary(self, operator, left, right):
        self.check(left)
        self.check(right)
        if isinstance(operator, ast.Add) and type(left) is type(right) and type(left) in (list, str):
            size = len(left) + len(right)
            if size > (self.limits.max_string_chars if type(left) is str else self.limits.max_container_items):
                self.fail_budget('Concatenation budget exhausted')
            self.allocate(chars=size) if type(left) is str else self.allocate(items=size)
            return self.data(left + right)
        if isinstance(operator, ast.Mult) and (type(left) in (list, str) or type(right) in (list, str)):
            value, count = (left, right) if type(left) in (list, str) else (right, left)
            if type(count) is not int:
                raise SkillError('Repeat count must be an integer')
            size = len(value) * max(0, count)
            if size > (self.limits.max_string_chars if type(value) is str else self.limits.max_container_items):
                self.fail_budget('Repetition budget exhausted')
            self.allocate(chars=size) if type(value) is str else self.allocate(items=size)
            return self.data(value * max(0, count))
        left, right = self.number(left), self.number(right)
        if isinstance(operator, ast.Pow):
            if type(left) is int and type(right) is int and right >= 0:
                if abs(left) in (0, 1):
                    result = 1 if right == 0 or left == 1 else (0 if left == 0 else (-1 if right % 2 else 1))
                else:
                    if right > self.limits.max_int_bits or (abs(left).bit_length() - 1) * right + 1 > self.limits.max_int_bits:
                        self.fail_budget('Power exceeds integer bit budget')
                    result = left ** right
            else:
                result = math.pow(left, right)
        elif isinstance(operator, ast.Add):
            result = left + right
        elif isinstance(operator, ast.Sub):
            result = left - right
        elif isinstance(operator, ast.Mult):
            if type(left) is int and type(right) is int and left and right and left.bit_length() + right.bit_length() > self.limits.max_int_bits + 1:
                self.fail_budget('Product exceeds integer bit budget')
            result = left * right
        elif isinstance(operator, ast.Div):
            result = left / right
        elif isinstance(operator, ast.FloorDiv):
            result = left // right
        elif isinstance(operator, ast.Mod):
            result = left % right
        else:
            raise SkillError('Unknown arithmetic operator')
        return self.data(result)

    def evaluate(self, node, local):
        self.tick()
        self.expression_depth += 1
        try:
            if self.expression_depth > 64:
                self.fail_budget('Expression/call nesting exceeds 64 levels')
            return self._evaluate(node, local)
        except SkillError:
            raise
        except (ArithmeticError, TypeError, KeyError, IndexError, ValueError, RuntimeError) as exc:
            raise SkillError(type(exc).__name__ + ': ' + str(exc)[:512]) from exc
        finally:
            self.expression_depth -= 1

    def _evaluate(self, node, local):
        if isinstance(node, ast.Constant):
            if type(node.value) is str:
                self.allocate(chars=len(node.value))
            return self.data(node.value)
        if isinstance(node, ast.Name):
            return self.lookup(node.id, local, self.globals)
        if isinstance(node, ast.List):
            self.allocate(items=len(node.elts))
            return self.data([self.evaluate(x, local) for x in node.elts])
        if isinstance(node, ast.Dict):
            self.allocate(items=len(node.keys))
            return self.data({self.evaluate(k, local): self.evaluate(v, local)
                              for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.evaluate(node.left, local), self.evaluate(node.right, local))
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand, local)
            if isinstance(node.op, ast.Not):
                return not self.data(value)
            return self.data(-self.number(value) if isinstance(node.op, ast.USub) else +self.number(value))
        if isinstance(node, ast.BoolOp):
            value = None
            for child in node.values:
                value = self.evaluate(child, local)
                self.data(value)
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
            return value
        if isinstance(node, ast.Compare):
            left = self.evaluate(node.left, local)
            for operator, child in zip(node.ops, node.comparators):
                right = self.evaluate(child, local)
                self.data(left)
                self.data(right)
                if isinstance(operator, ast.Eq): result = left == right
                elif isinstance(operator, ast.NotEq): result = left != right
                elif isinstance(operator, ast.Lt): result = left < right
                elif isinstance(operator, ast.LtE): result = left <= right
                elif isinstance(operator, ast.Gt): result = left > right
                elif isinstance(operator, ast.GtE): result = left >= right
                elif isinstance(operator, ast.In): result = left in right
                elif isinstance(operator, ast.NotIn): result = left not in right
                elif isinstance(operator, ast.Is): result = left is right
                else: result = left is not right
                if not result:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.evaluate(node.body if self.data(self.evaluate(node.test, local)) else node.orelse, local)
        if isinstance(node, ast.Subscript):
            value = self.data(self.evaluate(node.value, local))
            if type(value) not in (list, dict, str):
                raise SkillError('Only JSON containers and strings support subscripts')
            if isinstance(node.slice, ast.Slice):
                if type(value) not in (list, str):
                    raise SkillError('Only lists and strings support slices')
                parts = [self.evaluate(x, local) if x is not None else None
                         for x in (node.slice.lower, node.slice.upper, node.slice.step)]
                if any(x is not None and type(x) is not int for x in parts):
                    raise SkillError('Slice components must be integers')
                result = value[slice(*parts)]
                self.allocate(chars=len(result)) if type(result) is str else self.allocate(items=len(result))
                return self.data(result)
            key = self.evaluate(node.slice, local)
            if type(key) is not (str if type(value) is dict else int):
                raise SkillError('Invalid subscript type')
            return value[key]
        if isinstance(node, ast.Attribute):
            value = self.evaluate(node.value, local)
            if node.value.id == 'robot' and isinstance(value, _Robot):
                return _Callable('tool', node.attr)
            if node.value.id == 'math' and isinstance(value, _Math):
                return math.pi if node.attr == 'pi' else _Callable('math', node.attr)
            raise SkillError('Invalid capability receiver')
        if isinstance(node, ast.Call):
            function = self.evaluate(node.func, local)
            args = [self.evaluate(arg, local) for arg in node.args]
            keywords = {k.arg: self.evaluate(k.value, local) for k in node.keywords}
            return self.call(function, args, keywords)
        raise SkillError('Unsupported expression')

    def call(self, function, args, keywords):
        self.tick()
        if isinstance(function, _Function):
            names = [arg.arg for arg in function.node.args.args]
            bound = self.bind(names, args, keywords)
            self.calls += 1
            try:
                if self.calls > self.limits.max_call_depth:
                    self.fail_budget('Function call depth exhausted')
                try:
                    self.block(function.node.body, bound)
                except _Flow as flow:
                    if flow.kind == 'return':
                        return flow.value
                    raise SkillError('Invalid function control flow')
                return None
            finally:
                self.calls -= 1
        if not isinstance(function, _Callable):
            raise SkillError('Value is not a permitted callable')
        if function.kind == 'tool':
            return self.tool(function.name, args, keywords)
        if function.kind == 'math':
            if keywords or not args or len(args) > 16:
                raise SkillError('Math takes 1..16 positional numeric arguments')
            return self.data(_MATH[function.name](*[self.number(v) for v in args]))
        return self.builtin(function.name, args, keywords)

    @staticmethod
    def bind(names, args, keywords):
        if len(args) > len(names) or any(k not in names or k in names[:len(args)] for k in keywords):
            raise SkillError('Invalid call arguments')
        values = dict(zip(names, args))
        values.update(keywords)
        if set(values) != set(names):
            raise SkillError('Missing call argument')
        return values

    def tool(self, name, args, keywords):
        self.tick()
        values = self.bind(self.tool_specs[name], args, keywords)
        if name == 'move_tcp':
            for field, size in (('xyz_m', 3), ('quaternion_xyzw', 4)):
                value = values[field]
                if type(value) is not list or len(value) != size:
                    raise SkillError('Invalid tool vector: ' + field)
                for number in value:
                    self.number(number)
            if abs(math.hypot(*values['quaternion_xyzw']) - 1) > 1e-6:
                raise SkillError('Tool quaternion must be unit length')
        elif name == 'wait_bounded' and self.number(values['seconds']) < 0:
            raise SkillError('Wait must be nonnegative')
        elif name in ('play_sequence', 'preview_sequence'):
            if (type(values['keys']) is not list or not values['keys']
                    or any(type(key) is not str or not 1 <= len(key) <= 96
                           or not key.isascii() or not key.isalnum() for key in values['keys'])):
                raise SkillError('Sequence keys must be a nonempty JSON list of catalog key names')
            if type(values['options']) is not dict:
                raise SkillError('Sequence options must be a JSON dictionary')
        elif name in _DEMO_TOOLS and name != 'trajectory_catalog':
            identifier = values['trajectory_id']
            if (type(identifier) is not str or not 1 <= len(identifier) <= 96
                    or not identifier.isascii()
                    or not all(c.isalnum() or c in '_-' for c in identifier)):
                raise SkillError('Invalid trajectory identifier')
            if name == 'inspect_trajectory':
                if (type(values['start_index']) is not int or values['start_index'] < 0
                        or type(values['count']) is not int or not 1 <= values['count'] <= 32):
                    raise SkillError('Trajectory page requires nonnegative start and count 1..32')
            elif type(values['options']) is not dict:
                raise SkillError('Trajectory options must be a JSON dictionary')
        if name not in self.tools:
            raise SkillError('Tool capability unavailable: ' + name)
        if self.limits.max_tool_calls is not None and self.tool_calls >= self.limits.max_tool_calls:
            self.fail_budget('Tool call budget exhausted')
        self.tool_calls += 1
        values = self.clone(values)
        try:
            result = self.tools[name](**values)
        except ToolError:
            raise
        except SkillError as exc:
            self.fatal = exc
            raise
        except Exception as exc:
            self.fatal = SkillError('Host tool failed: ' + type(exc).__name__)
            raise self.fatal from exc
        try:
            return self.clone(result)
        except SkillError as exc:
            self.fatal = exc
            raise

    def builtin(self, name, args, keywords):
        if keywords:
            raise SkillError('Builtins only accept positional arguments')
        if name == 'str' and len(args) == 1 and isinstance(args[0], ToolError):
            value = str(args[0])
            self.allocate(chars=len(value))
            return self.data(value)
        if name == 'range':
            if not 1 <= len(args) <= 3 or any(type(v) is not int for v in args):
                raise SkillError('range takes 1..3 integers')
            for value in args:
                self.check(value)
            value = range(*args)
            try:
                if len(value) > self.limits.max_iterations:
                    self.fail_budget('Range exceeds iteration budget')
            except OverflowError:
                self.fail_budget('Range exceeds iteration budget')
            return value
        for value in args:
            if not (name == 'list' and isinstance(value, range)):
                self.data(value)
        if name in ('min', 'max', 'sum'):
            values = args[0] if len(args) == 1 and type(args[0]) is list else args
            if len(values) > self.limits.max_container_items:
                self.fail_budget('Reduction input exceeds budget')
            if name == 'sum':
                if len(args) != 1 or type(args[0]) is not list:
                    raise SkillError('sum requires one list')
                result = 0
                for value in values:
                    self.tick()
                    result = self.binary(ast.Add(), result, value)
                return result
            if not values:
                raise SkillError('Empty reduction')
            return self.data(min(values) if name == 'min' else max(values))
        if name in ('list', 'dict'):
            if not args:
                value = [] if name == 'list' else {}
            elif len(args) == 1 and ((name == 'list' and type(args[0]) in (list, range, str))
                                     or (name == 'dict' and type(args[0]) is dict)):
                if len(args[0]) > self.limits.max_container_items:
                    self.fail_budget('Container construction exceeds budget')
                value = list(args[0]) if name == 'list' else dict(args[0])
            else:
                raise SkillError('Unsupported container construction')
            return self.clone(value)
        if name == 'len' and len(args) == 1 and type(args[0]) in (list, dict, str):
            return len(args[0])
        if name == 'abs' and len(args) == 1:
            return self.data(abs(self.number(args[0])))
        if name == 'round' and 1 <= len(args) <= 2:
            self.number(args[0])
            if len(args) == 2 and (type(args[1]) is not int or abs(args[1]) > 128):
                raise SkillError('round precision must be an integer within 128')
            return self.data(round(*args))
        if name in ('float', 'int', 'bool', 'str') and len(args) <= 1:
            value = args[0] if args else ('' if name == 'str' else 0)
            if type(value) in (list, dict):
                if name == 'bool': return bool(value)
                raise SkillError('Container scalar conversion is not supported')
            if name == 'int': result = int(value)
            elif name == 'float': result = float(value)
            elif name == 'bool': result = bool(value)
            else:
                result = str(value)
                self.allocate(chars=len(result))
            return self.data(result)
        raise SkillError('Unsupported builtin arguments: ' + name)

    def assign(self, target, value, local):
        if not isinstance(value, (_Function, _Callable, _Robot, _Math, range)):
            self.check(value)
        if isinstance(target, ast.Name):
            local[target.id] = value
            return
        container = self.data(self.evaluate(target.value, local))
        key = self.evaluate(target.slice, local)
        if type(container) is dict and type(key) is str:
            if key not in container:
                self.allocate(items=1, chars=len(key))
        elif type(container) is list and type(key) is int:
            pass
        else:
            raise SkillError('Invalid subscript assignment')
        container[key] = value
        self.check(container)

    def block(self, statements, local):
        for statement in statements:
            self.statement(statement, local)

    def statement(self, node, local):
        self.tick()
        try:
            self._statement(node, local)
        except SkillError:
            raise
        except (ArithmeticError, TypeError, KeyError, IndexError, ValueError, RuntimeError) as exc:
            raise SkillError(type(exc).__name__ + ': ' + str(exc)[:512]) from exc

    def _statement(self, node, local):
        if isinstance(node, ast.Assign):
            value = self.evaluate(node.value, local)
            for target in node.targets:
                self.assign(target, value, local)
        elif isinstance(node, ast.AugAssign):
            value = self.binary(node.op, self.evaluate(node.target, local), self.evaluate(node.value, local))
            self.assign(node.target, value, local)
        elif isinstance(node, ast.Expr):
            self.evaluate(node.value, local)
        elif isinstance(node, ast.If):
            self.block(node.body if self.data(self.evaluate(node.test, local)) else node.orelse, local)
        elif isinstance(node, (ast.While, ast.For)):
            iterable = None
            if isinstance(node, ast.For):
                value = self.evaluate(node.iter, local)
                if type(value) not in (list, dict, str, range):
                    raise SkillError('Unsupported loop iterable')
                iterable = iter(value)
            broken = False
            while True:
                self.tick()
                if iterable is None:
                    if not self.data(self.evaluate(node.test, local)):
                        break
                else:
                    try:
                        item = next(iterable)
                    except StopIteration:
                        break
                    self.assign(node.target, item, local)
                self.iterations += 1
                if self.iterations > self.limits.max_iterations:
                    self.fail_budget('Loop iteration budget exhausted')
                try:
                    self.block(node.body, local)
                except _Flow as flow:
                    if flow.kind == 'break':
                        broken = True
                        break
                    if flow.kind != 'continue':
                        raise
            if not broken:
                self.block(node.orelse, local)
        elif isinstance(node, ast.Return):
            raise _Flow('return', self.evaluate(node.value, local) if node.value else None)
        elif isinstance(node, ast.Break):
            raise _Flow('break')
        elif isinstance(node, ast.Continue):
            raise _Flow('continue')
        elif isinstance(node, ast.Try):
            pending = None
            try:
                try:
                    self.block(node.body, local)
                except ToolError as exc:
                    if not node.handlers:
                        raise
                    handler = node.handlers[0]
                    if handler.name:
                        local[handler.name] = exc
                    try:
                        self.block(handler.body, local)
                    finally:
                        if handler.name:
                            local.pop(handler.name, None)
                else:
                    self.block(node.orelse, local)
            except BaseException as exc:
                pending = exc
            try:
                self.block(node.finalbody, local)
            except BaseException as exc:
                if not isinstance(pending, SkillError) or isinstance(exc, BudgetExceeded):
                    pending = exc
            if pending is not None:
                raise pending
        elif not isinstance(node, ast.Pass):
            raise SkillError('Unsupported statement')


def run_skill(sources, *, tools, profile, limits=Limits(), demo_tools=False):
    """Validate every source, initialize a clean namespace, then call entry main.

    ``tools`` callbacks accept the corresponding keyword arguments: observe(),
    move_tcp(xyz_m=..., quaternion_xyzw=...), wait_bounded(seconds=...), stop().
    Only an explicit ToolError is recoverable. Other callback failures latch
    execution closed, including finally, so transport loss cannot trigger moves.
    """
    if not isinstance(limits, Limits):
        raise ValidationError('Explicit Limits instance required')
    if type(sources) is not list or not sources or len(sources) > limits.max_container_items:
        raise ValidationError('Nonempty ordered source list required')
    allowed_tools = tool_specs(demo_tools=demo_tools)
    if type(tools) is not dict or set(tools) - set(allowed_tools) or any(not callable(f) for f in tools.values()):
        raise ValidationError('Only named robot callbacks are allowed')
    if type(profile) is not dict:
        raise ValidationError('Profile must be a JSON dictionary')
    seen, total_bytes = set(), 0
    for source in sources:
        if (type(source) is not tuple or len(source) != 2 or type(source[0]) is not str
                or not source[0] or '\x00' in source[0] or source[0] in seen or type(source[1]) is not str):
            raise ValidationError('Sources must have unique filenames and text')
        seen.add(source[0])
        try:
            total_bytes += len(source[1].encode('utf-8'))
        except UnicodeError as exc:
            raise ValidationError('Invalid source Unicode') from exc
    if total_bytes > limits.max_source_bytes:
        raise ValidationError('Source byte budget exceeded')
    trees, count, names = [], 0, set()
    for filename, source in sources:
        try:
            tree = ast.parse(source, filename=filename)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValidationError(filename + ':' + str(getattr(exc, 'lineno', 1)) + ': invalid source') from exc
        count += sum(1 for _ in ast.walk(tree))
        if count > limits.max_ast_nodes:
            raise ValidationError('AST node budget exceeded')
        trees.append(tree)
    callable_names = _BUILTINS | {n.name for tree in trees for n in tree.body if isinstance(n, ast.FunctionDef)}
    for (filename, _), tree in zip(sources, trees):
        try:
            _validate(tree, limits, callable_names, demo_tools=demo_tools)
        except ValidationError as exc:
            raise ValidationError(filename + ': ' + str(exc)) from exc
        for statement in tree.body:
            if isinstance(statement, ast.FunctionDef):
                name = statement.name
            elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                  and isinstance(statement.targets[0], ast.Name) and _constant_expression(statement.value)):
                name = statement.targets[0].id
            else:
                raise ValidationError('Top level permits only functions and literal constant expressions')
            if name in names or name in _RESERVED or name in {'robot', 'profile'}:
                raise ValidationError('Duplicate or reserved module name: ' + name)
            names.add(name)
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'main' for n in trees[-1].body):
        raise ValidationError('Last source must define main(robot, profile)')
    interpreter = _Interpreter(tools, limits, demo_tools=demo_tools)
    profile = interpreter.clone(profile)
    for tree in trees:
        for statement in tree.body:
            if isinstance(statement, ast.FunctionDef):
                interpreter.globals[statement.name] = _Function(statement)
            else:
                interpreter.statement(statement, interpreter.globals)
    main = interpreter.globals['main']
    if not isinstance(main, _Function) or len(main.node.args.args) != 2:
        raise ValidationError('main must have exactly two parameters')
    try:
        result = interpreter.call(main, [_Robot(), profile], {})
        _, characters = interpreter.check(result)
        if characters > 32768 or len(json.dumps(result, ensure_ascii=False, allow_nan=False,
                                                separators=(',', ':')).encode('utf-8')) > 32768:
            interpreter.fail_budget('Serialized result exceeds 32 KiB')
        return result
    except RecursionError as exc:
        raise BudgetExceeded('Host recursion guard reached') from exc
