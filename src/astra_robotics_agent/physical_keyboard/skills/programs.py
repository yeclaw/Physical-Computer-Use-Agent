"""Model-authored skill interfaces and mechanical, bounded program linking.

Contracts describe the model's claims; they do not prove physical preconditions,
postconditions, or success. This module validates finite data and the existing
skill language only. It never executes a candidate or obtains robot authority.
Dependency aliases are ordinary internal helpers. Their dynamic arguments are
part of the parent program's semantics, not a statically proven schema call.
"""
from __future__ import annotations

import ast
import hashlib
import json
import keyword
import math

from astra_robotics_agent.physical_keyboard.runtime import skill_language
from astra_robotics_agent.physical_keyboard.skills.knowledge import (
    KnowledgeError, _profile, _validate_source,
)


class ProgramError(ValueError):
    """A declaration or linked program is invalid before any dispatch."""


_TYPES = {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}
_COMMON_SCHEMA = {'type', 'enum', 'description'}
_SCHEMA_FIELDS = {
    'object': {'properties', 'required', 'additionalProperties'},
    'array': {'items', 'minItems', 'maxItems'},
    'string': {'minLength', 'maxLength'},
    'integer': {'minimum', 'maximum'},
    'number': {'minimum', 'maximum'},
    'boolean': set(), 'null': set(),
}


def _json_copy(value, label):
    """Reject non-JSON/custom values and allocation-shaped inputs before copying."""
    count = 0

    def visit(item, depth=0, ancestors=frozenset()):
        nonlocal count
        count += 1
        if depth > 16 or count > 2048:
            raise ProgramError(label + ' exceeds the structural budget')
        kind = type(item)
        if kind in (dict, list):
            if id(item) in ancestors:
                raise ProgramError(label + ' contains a cycle')
            ancestors = ancestors | {id(item)}
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str or not key or len(key) > 128:
                        raise ProgramError(label + ' requires nonempty string keys of at most 128 characters')
                    visit(child, depth + 1, ancestors)
            else:
                for child in item:
                    visit(child, depth + 1, ancestors)
        elif kind is str:
            if len(item) > 4096:
                raise ProgramError(label + ' string exceeds 4096 characters')
        elif kind is int:
            if item.bit_length() > 128:
                raise ProgramError(label + ' integer exceeds 128 bits')
        elif kind is float:
            if not math.isfinite(item):
                raise ProgramError(label + ' must contain finite numbers')
        elif kind not in (bool, type(None)):
            raise ProgramError(label + ' must contain only plain JSON values')

    visit(value)
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(raw.encode('utf-8')) > 65536:
            raise ProgramError(label + ' exceeds 64 KiB')
        return json.loads(raw)
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ProgramError):
            raise
        raise ProgramError(label + ' is not valid finite UTF-8 JSON') from exc


def _text(value, label):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ProgramError(label + ' must be a nonempty string of at most 4096 characters')


def _texts(value, label):
    if type(value) is not list or len(value) > 32:
        raise ProgramError(label + ' must contain at most 32 strings')
    for item in value:
        _text(item, label)


def _equal(left, right):
    # Python equates True with 1; JSON-schema enum matching must not do that.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (set(left) == set(right)
                and all(_equal(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _schema(schema, label='input_schema', depth=0):
    if (depth > 12 or type(schema) is not dict or type(schema.get('type')) is not str
            or schema.get('type') not in _TYPES):
        raise ProgramError(label + ' requires a supported explicit type within 12 schema levels')
    kind = schema['type']
    unknown = set(schema) - _COMMON_SCHEMA - _SCHEMA_FIELDS[kind]
    if unknown:
        raise ProgramError(label + ' has unsupported keywords: ' + ', '.join(sorted(unknown)))
    if 'description' in schema:
        _text(schema['description'], label + '.description')
    if 'enum' in schema:
        values = schema['enum']
        if type(values) is not list or not 1 <= len(values) <= 128:
            raise ProgramError(label + '.enum requires 1..128 values')
        if any(_equal(value, previous) for i, value in enumerate(values) for previous in values[:i]):
            raise ProgramError(label + '.enum must contain unique JSON values')
    if kind == 'object':
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        if type(properties) is not dict or len(properties) > 128:
            raise ProgramError(label + '.properties must be an object of at most 128 properties')
        if (type(required) is not list or any(type(key) is not str for key in required)
                or len(set(required)) != len(required) or not set(required) <= set(properties)):
            raise ProgramError(label + '.required must list unique declared properties')
        if type(schema.get('additionalProperties', False)) is not bool:
            raise ProgramError(label + '.additionalProperties must be boolean')
        for key, child in properties.items():
            _schema(child, label + '.properties.' + key, depth + 1)
    elif kind == 'array':
        if 'items' not in schema:
            raise ProgramError(label + '.items is required for arrays')
        _schema(schema['items'], label + '.items', depth + 1)
        _integer_bounds(schema, label, 'minItems', 'maxItems', 1024)
    elif kind == 'string':
        _integer_bounds(schema, label, 'minLength', 'maxLength', 4096)
    elif kind in ('integer', 'number'):
        for key in ('minimum', 'maximum'):
            if key in schema and type(schema[key]) not in (int, float):
                raise ProgramError(label + '.' + key + ' must be a finite number')
        if ('minimum' in schema and 'maximum' in schema
                and schema['minimum'] > schema['maximum']):
            raise ProgramError(label + ': minimum exceeds maximum')


def _integer_bounds(schema, label, lower, upper, maximum):
    for key in (lower, upper):
        if key in schema and (type(schema[key]) is not int or not 0 <= schema[key] <= maximum):
            raise ProgramError(label + '.' + key + ' must be an integer in 0..' + str(maximum))
    if schema.get(lower, 0) > schema.get(upper, maximum):
        raise ProgramError(label + ': lower length bound exceeds upper bound')


def validate_contract(contract):
    """Validate passive model declarations; input-schema defaults to strict objects.

    Omitted additionalProperties means false. This is a deliberately limited
    schema subset, not a general JSON Schema implementation. Preconditions,
    postconditions and side effects remain model declarations tied to evidence.
    """
    result = _json_copy(contract, 'contract')
    required = {'description', 'input_schema', 'preconditions', 'postconditions'}
    if type(result) is not dict or not required <= set(result) or set(result) - required - {'side_effects'}:
        raise ProgramError('contract requires description/input_schema/preconditions/postconditions and optional side_effects')
    _text(result['description'], 'description')
    for name in ('preconditions', 'postconditions', 'side_effects'):
        if name in result:
            _texts(result[name], name)
    _schema(result['input_schema'])
    if result['input_schema']['type'] != 'object':
        raise ProgramError('input_schema must describe an object')
    return result


def _matches(value, schema, path):
    kind = schema['type']
    valid = {
        'object': type(value) is dict, 'array': type(value) is list,
        'string': type(value) is str, 'integer': type(value) is int,
        'number': type(value) in (int, float), 'boolean': type(value) is bool,
        'null': value is None,
    }[kind]
    if not valid:
        raise ProgramError(path + ' must have type ' + kind)
    if 'enum' in schema and not any(_equal(value, candidate) for candidate in schema['enum']):
        raise ProgramError(path + ' does not match its enum')
    if kind == 'object':
        properties = schema.get('properties', {})
        missing = set(schema.get('required', [])) - set(value)
        if missing:
            raise ProgramError(path + ' is missing required properties: ' + ', '.join(sorted(missing)))
        if not schema.get('additionalProperties', False) and set(value) - set(properties):
            raise ProgramError(path + ' has undeclared properties')
        for key in value:
            if key in properties:
                _matches(value[key], properties[key], path + '.' + key)
    elif kind in ('array', 'string'):
        lower, upper = ('minItems', 'maxItems') if kind == 'array' else ('minLength', 'maxLength')
        if len(value) < schema.get(lower, 0) or len(value) > schema.get(upper, 1024 if kind == 'array' else 4096):
            raise ProgramError(path + ' violates its length bounds')
        if kind == 'array':
            for i, item in enumerate(value):
                _matches(item, schema['items'], path + '[' + str(i) + ']')
    elif kind in ('integer', 'number'):
        if ('minimum' in schema and value < schema['minimum']
                or 'maximum' in schema and value > schema['maximum']):
            raise ProgramError(path + ' violates its numeric bounds')


def validate_arguments(contract, arguments):
    """Return a finite copy of declared external call arguments, without execution."""
    contract = validate_contract(contract)
    arguments = _json_copy(arguments, 'arguments')
    if type(arguments) is not dict:
        raise ProgramError('arguments must be an object')
    _matches(arguments, contract['input_schema'], 'arguments')
    return arguments


def _source(lines):
    if (type(lines) is not list or not 1 <= len(lines) <= 1024
            or any(type(line) is not str or '\n' in line or '\r' in line for line in lines)):
        raise ProgramError('source_lines must contain 1..1024 individual Python lines')
    return '\n'.join(lines) + '\n'


def _symbols(tree):
    return [node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
            for node in tree.body]


class _Namespace(ast.NodeTransformer):
    """Rename globals and same-spelled local bindings coherently in the subset."""

    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self.names.get(node.arg, node.arg)
        return node

    def visit_keyword(self, node):
        # Function keyword names that happen to match renamed symbols must
        # follow their corresponding renamed formal parameter. Robot keyword
        # names are handled by visit_Call and remain the fixed robot interface.
        node.value = self.visit(node.value)
        return node

    def visit_Call(self, node):
        named = isinstance(node.func, ast.Name) and node.func.id in self.names
        node.func = self.visit(node.func)
        node.args = [self.visit(value) for value in node.args]
        node.keywords = [self.visit(value) for value in node.keywords]
        if named:
            for argument in node.keywords:
                argument.arg = self.names.get(argument.arg, argument.arg)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)


def compile_program(source_lines, profile, dependencies=None, *, demo_tools=False):
    """Link declared frozen helpers into one program for the existing executor.

    Each alias(robot, parameters) starts with a fresh dependency default profile
    and shallowly overlays parameters. The parent model chooses which aliases
    to call, in what order, and whether a returned result allows continuation.
    No calls happen here; source validation retains the active tool whitelist.
    """
    if type(demo_tools) is not bool:
        raise ProgramError('demo_tools must be an explicit boolean')
    source = _source(source_lines)
    try:
        copied_profile = json.loads(_profile(profile))
        dependencies = {} if dependencies is None else dependencies
        if type(dependencies) is not dict or len(dependencies) > 8:
            raise ProgramError('dependencies must be an alias map of at most eight entries')
        for alias in dependencies:
            if (type(alias) is not str or not alias.isidentifier() or alias.startswith('_')
                    or keyword.iskeyword(alias) or len(alias) > 64
                    or alias in skill_language._RESERVED | skill_language._FORBIDDEN | {'main', 'robot', 'profile'}):
                raise ProgramError('Dependency alias must be a nonreserved public identifier')
        aliases = sorted(dependencies)
        root = ast.parse(source, filename='model-skill.py')
        # Temporary declarations validate unresolved aliases without changing
        # the model's program. They are replaced by the frozen helpers below.
        stubs = ''.join('def ' + alias + '(robot, parameters):\n    return None\n' for alias in aliases)
        _validate_source(stubs + source, demo_tools=demo_tools)
        root_symbols = set(_symbols(root))
        if root_symbols & set(aliases):
            raise ProgramError('Dependency alias conflicts with a parent declaration')
        occupied = root_symbols | set(aliases)
        linked = []
        pins = []
        for alias in aliases:
            dependency = dependencies[alias]
            if (type(dependency) is not dict or not {'source', 'profile', 'candidate_id'} <= set(dependency)
                    or type(dependency['source']) is not str
                    or type(dependency['candidate_id']) is not str or not dependency['candidate_id']
                    or len(dependency['candidate_id']) > 256):
                raise ProgramError('Dependency must have frozen source/profile/candidate_id')
            child_source = dependency['source']
            child_profile_raw = _profile(dependency['profile'])
            child_profile = json.loads(child_profile_raw)
            _validate_source(child_source, demo_tools=demo_tools)
            child = ast.parse(child_source, filename='dependency-' + alias + '.py')
            source_sha = hashlib.sha256(child_source.encode('utf-8')).hexdigest()
            profile_sha = hashlib.sha256(child_profile_raw).hexdigest()
            pin = dict(alias=alias, candidate_id=dependency['candidate_id'],
                       source_sha256=source_sha, profile_sha256=profile_sha)
            prefix = 'linked' + hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()[:12] + 'x'
            names = {name: prefix + name for name in _symbols(child)}
            if occupied & set(names.values()):
                raise ProgramError('Generated dependency namespace conflicts with another declaration')
            occupied.update(names.values())
            child = _Namespace(names).visit(child)
            linked.extend(child.body)
            wrapper = (
                'def ' + alias + '(robot, parameters):\n'
                '    dependencyProfile = ' + repr(child_profile) + '\n'
                '    for parameterName in parameters:\n'
                '        dependencyProfile[parameterName] = parameters[parameterName]\n'
                '    return ' + names['main'] + '(robot, dependencyProfile)\n'
            )
            linked.extend(ast.parse(wrapper).body)
            pins.append(pin)
        linked.extend(root.body)
        compiled = ast.unparse(ast.fix_missing_locations(ast.Module(body=linked, type_ignores=[]))) + '\n'
        _validate_source(compiled, demo_tools=demo_tools)
        functions = [node for node in root.body if isinstance(node, ast.FunctionDef)]
        main = next(node for node in functions if node.name == 'main')
        return dict(
            source_lines=compiled.rstrip('\n').split('\n'), profile=copied_profile,
            entrypoint=dict(name='main', parameters=[arg.arg for arg in main.args.args],
                docstring=ast.get_docstring(main, clean=False),
                functions=[dict(name=node.name, parameters=[arg.arg for arg in node.args.args],
                                docstring=ast.get_docstring(node, clean=False)) for node in functions],
                dependency_aliases=aliases),
            dependency_pins=pins,
        )
    except (KnowledgeError, SyntaxError, UnicodeError, RecursionError) as exc:
        raise ProgramError('Skill linking rejected: ' + str(exc)) from exc
