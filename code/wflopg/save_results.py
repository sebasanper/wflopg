from collections import OrderedDict
from uuid import uuid4 as uuid
from ruamel.yaml import YAML as yaml


def layout2yaml(layout, site, name, filename):
    output = {}  # OrderedDict()
    output['name'] = name
    output['uuid'] = str(uuid())
    output['site'] = site
    output['layout'] = layout.values.tolist()
    with open(filename, 'w') as f:
        yaml(typ='safe').dump(output, f)
