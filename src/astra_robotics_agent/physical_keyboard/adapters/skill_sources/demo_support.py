"""Small optional model context; importing this module reads no demo or SDK."""
import re


DEMO_TRAJECTORY_HINTS = ""


def normalize_demo_capabilities(value):
    if type(value) is not dict or set(value) != {'available', 'timed_replay_available',
                                               'catalog_sha256', 'trajectory_count'}:
        raise ValueError('exact optional demonstration capability descriptor required')
    if value['available'] is not True or type(value['timed_replay_available']) is not bool:
        raise ValueError('configured demonstration availability must be explicit')
    if type(value['catalog_sha256']) is not str or not re.fullmatch(r'[a-f0-9]{64}', value['catalog_sha256']):
        raise ValueError('catalog SHA256 required')
    if type(value['trajectory_count']) is not int or value['trajectory_count'] < 1:
        raise ValueError('positive trajectory count required')
    return dict(value)
