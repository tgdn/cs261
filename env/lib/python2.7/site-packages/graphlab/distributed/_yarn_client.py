import requests as _requests
import os
from graphlab.util import subprocess_exe


def _get_default_conf_dir():
    from graphlab import sys_util
    classpath = sys_util.get_hadoop_class_path()
    if classpath:
        return classpath.split(':')[0]
    else:
        return "/etc/hadoop/conf"


def _parse(xml, key):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml)
    root = tree.getroot()
    properties = root.findall('./property')
    name_value_pairs = [dict((e.tag, e.text) for e in p) for p in properties]
    name_value_dict = dict((x['name'], x['value']) for x in name_value_pairs)
    value = name_value_dict.get(key, None)
    if value:
        try:
            value = _var_substitute(value, name_value_dict)
        except KeyError as e:
            raise KeyError('Unable to parse the value of Key: %s. Value substitution failed with missing key %s' % (key, str(e)))
    return value


def _var_substitute(value, name_value_dict):
    """
    Substitute variable in the the form of ${...} with its value in the given dictionary

    Example:
    -------
    >>> value = '${yarn.resourcemanager.hostname}:${yarn.resourcemanager.port}'
    >>> d = {"yarn.resourcemanager.hostname": "localhost", "yarn.resourcemanager.port": "1234"}
    >>> _var_substitute(value, d)
    localhost:1234
    """
    import re
    var_pattern = '\$\{([a-z.]+)\}'
    variables = re.findall(var_pattern, value)
    if len(variables) == 0:
        return value

    subs = [name_value_dict[x] for x in variables]
    var_pattern_nocapture = '\$\{[a-z.]+\}'
    for i in range(len(variables)):
        value = re.sub(var_pattern_nocapture, subs[i], value, count=1)
    return value


def get_yarn_conf_key(key_name, conf_dir=None):
    if conf_dir is None:
        conf_dir = _get_default_conf_dir()
    config_path = os.path.join(conf_dir, 'yarn-site.xml')
    if not os.path.exists(config_path):
        raise IOError("Cannot find yarn configuration file at %s" % config_path)
    return _parse(config_path, key_name)


class YarnClient(object):
    def __init__(self, env, rest_version="v1"):
        self.conf_dir = env.hadoop_conf_dir

        self.rm_endpoint = self._get_rm_rest_endpoint()
        self.nm_endpoint = None
        self.version = rest_version

    def _get_rm_rest_endpoint(self):
        protocol = ""
        rmid = ""

        # Check for HTTP or HTTPS
        http_pol = get_yarn_conf_key("yarn.http.policy", self.conf_dir)
        if http_pol and http_pol[0:10] == "HTTPS_ONLY":
            protocol = ".https"
            self.protocol = "https"
        else:
            # Don't need to fill 'protocol' in this case, it is blank if http
            self.protocol = "http"

        # If HA is enabled, choose the active endpoint
        ha_enabled = get_yarn_conf_key("yarn.resourcemanager.ha.enabled", self.conf_dir)
        if ha_enabled and ha_enabled[0:4] == "true":
            rm_ids = get_yarn_conf_key("yarn.resourcemanager.ha.rm-ids", self.conf_dir)
            rm_list = rm_ids.split(",")
            for i in rm_list:
                sub_ret = subprocess_exe("yarn", ["rmadmin", "-getServiceState", i])
                if not sub_ret['success']:
                    raise RuntimeError(sub_ret['stderr'])
                if sub_ret['stdout'][0:6] == "active":
                    rmid = "." + i
                    break

        endpoint_key = "yarn.resourcemanager.webapp" + protocol + ".address" + rmid
        endpoint_url = get_yarn_conf_key(endpoint_key, self.conf_dir)

        if not endpoint_url:
            raise RuntimeError('Unable to parse resource manager address')

        return endpoint_url

    # Uses YARN's "Cluster New Application API" and "Cluster Applications API"
    # from the ResourceManager REST API. These APIs have been available since
    # Hadoop 2.5.
    # TODO: Skipping the full implementation for now. I think this is how we
    # want to launch applications in the future.
    def submit_application(self, jar_file):

        r = _requests.post('/'.join([self.protocol + '://' + self.rm_endpoint,
                                    "ws",
                                    self.rest_version,
                                    "cluster",
                                    "apps",
                                    "new-application",
                                    ]))
        r.raise_for_status()

        resp = r.json()
        self.app_id = resp['application-id']
        self.max_resource = resp['maximum-response-capabilities']

        raise NotImplementedError


    def get_state(self, app_id):
        url = '/'.join([self.protocol + '://' + self.rm_endpoint,
                        "ws",
                        self.version,
                        "cluster",
                        "apps",
                        app_id,
                        "state",
                        ])
        r = _requests.get(url)

        r.raise_for_status()

        return r.json()

    def get_app_info(self, app_id):
        url = '/'.join([self.protocol + '://' + self.rm_endpoint,
                        "ws",
                        self.version,
                        "cluster",
                        "apps",
                        app_id,
                        ])
        r = _requests.get(url)

        r.raise_for_status()

        return r.json()

    def cancel(self, app_id):
        raise NotImplementedError
