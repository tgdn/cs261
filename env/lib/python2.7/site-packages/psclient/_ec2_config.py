class Ec2Config(object):
    '''
    Configuration object for launching EC2 clusters.

    Parameters
    ----------
    region : str, optional
        The AWS region in which to launch your instance.

    instance_type : str, optional
        The EC2 instance type to launch, default is m3.xlarge. We support all
        instance types except t1.micro, T2, c1.medium, m1.small, m1.medium,
        m3.medium, and m4.

    aws_access_key_id : str, optional
        The AWS Access Key to use to launch the host(s). This parameter must be
        set or the AWS_ACCESS_KEY_ID environment variable must be set.

    aws_secret_access_key : str, optional
        The AWS Secret Key to use to launch the host(s). This parameter must
        be set or the AWS_SECRET_ACCESS_KEY environment variable must be set.

    security_group : str, optional
        The name of the security group for the EC2 instance to use. The necessary
        rules will be added to this security group.

    tags : dict, optional
        A dictionary containing the name/value tag pairs to be assigned to the
        host(s). If you want to create only a tag name, the value for that
        tag should be the empty string (i.e. ''). In addition to these
        specified tags, a 'GraphLab' tag will also be assigned.

    cidr_ip : str | list [str], optional
        The Classless Inter-Domain Routing rule(s) to use for the instance.
        Useful for restricting the IP Address Range for a client. Default is
        no restriction. If you specify this, you must also specify a security
        group to use.

    subnet_id : str, optional
        The ID of the subnet for the EC2 instance to use.
        This is needed if you want to specify a particular subnet for the instance
        to launch in.

    security_group_id : str, optional
        The id of the security group for the EC2 instance to use. The necessary
        rules will be added to this security group.

    Returns
    -------
    ~psclient.Ec2Config
        An EC2 configuration object to be used by :func:`psclient.create`.

    Notes
    -----
    The security group name is commonly used with EC2-Classic or default
    VPC deployments. For a non-default VPC, use security_group_id in
    conjunction with subnet_id to specify your network security setting.

    A security group name is not unique among different VPC networks, so avoid
    using security group name if possible.

    The security_group_id is commonly used for a non-default VPC, where it
    is used in conjunction with a subnet_id to define the security policy
    for instances launched within the VPC.

    You cannot provide both security_group and security_group_id at the same time.

    If neither security_group nor security_group_id are specified, a default
    security group will be created. The name of the default security group
    is "Turi_Predictive_Service".

    See Also
    --------
    psclient.create
    '''
    def __init__(self, region = 'us-west-2', instance_type = 'm3.xlarge',
                 aws_access_key_id = None, aws_secret_access_key = None,
                 security_group = None, tags = None, cidr_ip = None,
                 subnet_id = None, security_group_id = None):

        #We do not allow both security_group_name and security_group_id to be specified.
        if security_group is not None and security_group_id is not None:
            raise Exception("Specify either security_group or security_group_id but not Both.")

        # region
        if isinstance(region, str):
            from aws import VALID_REGIONS
            if region not in VALID_REGIONS:
                raise TypeError("%s is not a valid AWS region." % region)
            self.region = region
        else:
            raise TypeError('\'region\' must be a string.')

        # instance_type
        if isinstance(instance_type, str):
            # TODO: Need better instance type checking, like we do for region
            self.instance_type = instance_type
        else:
            raise TypeError('\'instance_type\' must be a string.')

        # aws keys
        if bool(aws_access_key_id) != bool(aws_secret_access_key):
            raise RuntimeError('If you specify a \'aws_access_key_id\' or a \'aws_secret_access_key\','
                               ' you must specify both \'aws_access_key_id\' and'
                               ' \'aws_secret_access_key\'.')
        if aws_access_key_id and aws_secret_access_key:
            # aws keys where passed in
            if isinstance(aws_access_key_id, str) and isinstance(aws_secret_access_key, str):
                self.aws_access_key_id = aws_access_key_id
                self.aws_secret_access_key = aws_secret_access_key
            else:
                raise TypeError('\'aws_access_key_id\' and \'aws_secret_access_key\' must be string')
        else:
            # Try to get aws keys for the environment variables
            try:
                from aws import get_credentials
                self.aws_access_key_id, self.aws_secret_access_key = get_credentials()
            except:
                raise RuntimeError('AWS credentials must either be passed in or set as'
                                   ' environment variables.')

        # security_group
        if not security_group or isinstance(security_group, str):
            self.security_group = security_group
        else:
            raise TypeError('\'security_group\' must be a string.')

        # tags
        if not tags:
            self.tags = {}
        elif isinstance(tags, dict):
            self.tags = tags
        else:
            raise TypeError('\'tags\' must be a dict.')

        # cidr_ip
        if cidr_ip and not (isinstance(cidr_ip, str) or isinstance(cidr_ip, list)):
            raise TypeError('\'cidr_ip\' must be a string or a list of strings')
        else:
            self.cidr_ip = cidr_ip

        # subnet_id
        if not subnet_id or isinstance(subnet_id, str):
            self.subnet_id = subnet_id
        else:
            raise TypeError('\'subnet_id\' must be a string.')

        # security_group_id
        if not security_group_id or isinstance(security_group_id, str):
            self.security_group_id = security_group_id
        else:
            raise TypeError('\'security_group_id\' must be a string.')

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "[instance_type: %s, region: %s, aws_access_key: %s]" % \
            (self.instance_type, self.region, self.aws_access_key_id)

    def get_credentials(self):
        """
        Returns the AWS credentials as a dictionary with the following keys:

            - 'aws_access_key_id'       : aws_access_key
            - 'aws_secret_access_key'   : aws_secret_key

        Returns
        -------
        dict
            A dictionary containing the AWS credentials.

        """
        return {
            'aws_access_key_id': self.aws_access_key_id,
            'aws_secret_access_key': self.aws_secret_access_key
        }
