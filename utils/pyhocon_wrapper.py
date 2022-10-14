from typing import Sequence

import pyhocon


class ConfigTree:
    def __init__(self, config_tree: pyhocon.ConfigTree):
        self._pyhocon = config_tree or pyhocon.ConfigTree({})
        self._used_keys = set()

    def get_bool(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_bool(key, default)

    def get_int(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_int(key, default)

    def get_float(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_float(key, default)

    def get_string(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_string(key, default)

    def get_list(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_list(key, default)

    def get_config(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get_config(key, default)

    def pop_config(self, key, default=pyhocon.UndefinedKey):
        config = self.get_config(key, default=default)
        self._mark_config_as_used(config, key)
        return ConfigTree(config)

    def _mark_config_as_used(self, config, parent_key):
        if not config:
            return

        for key, value in config.items():
            if isinstance(value, pyhocon.ConfigTree):
                self._mark_config_as_used(value, f'{parent_key}.{key}')
            else:
                self._used_keys.add(f'{parent_key}.{key}')

    def get(self, key, default=pyhocon.UndefinedKey):
        self._used_keys.add(key)
        return self._pyhocon.get(key, default)

    def __contains__(self, item):
        return item in self._pyhocon

    def __iter__(self):
        return self._pyhocon.__iter__()

    def put(self, key, value):
        self._pyhocon.put(key, value)

    def validate(self):
        keys = {}
        _flatten_tree('', self._pyhocon, keys)
        keys = set(keys)
        unused_keys = keys.difference(self._used_keys)
        if unused_keys:
            raise UnknownParameters(', '.join(unused_keys))

    def get_impl(self):
        return self._pyhocon


class UnknownParameters(Exception):
    pass


def parse_string(string: str):
    return ConfigTree(pyhocon.ConfigFactory.parse_string(string))


def parse_file(path: str):
    return ConfigTree(pyhocon.ConfigFactory.parse_file(path))


def override_config(config: ConfigTree, override_list: Sequence[str]):
    for override in override_list:
        name_value = override.split('=')
        if len(name_value) != 2:
            raise Exception(f'Bad configuration override: {override}')
        config.get_impl()[name_value[0]] = name_value[1]


def dump_config(config: ConfigTree, path: str):
    with open(path, 'w') as f:
        f.write(pyhocon.HOCONConverter.to_hocon(config.get_impl(), indent=4))


def to_str(config: ConfigTree) -> str:
    return pyhocon.HOCONConverter.to_hocon(config.get_impl(), indent=4)


def _join_keys(parent_key: str, child_key: str):
    if parent_key and child_key:
        return parent_key + '.' + child_key
    else:
        return parent_key + child_key


def _flatten_tree(config_key, config_value, result_dict):
    if isinstance(config_value, pyhocon.ConfigTree):
        for key, value in config_value.items():
            _flatten_tree(_join_keys(config_key, key), value, result_dict)
    else:
        result_dict[config_key] = config_value
