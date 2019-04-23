from collections import OrderedDict as _odict
from uuid import uuid4 as _uuid
from ruamel.yaml import YAML as _yaml


def layout2yaml(layout, site, name, filename):
    output = {}  # _odict()
    output['name'] = name
    output['uuid'] = str(_uuid())
    output['site'] = site
    output['layout'] = layout.values.tolist()
    with open(filename, 'w') as f:
        _yaml(typ='safe').dump(output, f)
