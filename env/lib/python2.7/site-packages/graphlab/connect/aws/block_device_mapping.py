from boto.ec2.blockdevicemapping import BlockDeviceMapping, BlockDeviceType
import string

# Source: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/InstanceStorage.html
# For easy maintainability, please keep this manually-inputted dictionary sorted by key.
_EC2_INSTANCE_NUM_DISKS = {
    "c1.medium":   1,
    "c1.xlarge":   4,
    "c3.large":    2,
    "c3.xlarge":   2,
    "c3.2xlarge":  2,
    "c3.4xlarge":  2,
    "c3.8xlarge":  2,
    "c4.large":    0,
    "c4.xlarge":   0,
    "c4.2xlarge":  0,
    "c4.4xlarge":  0,
    "c4.8xlarge":  0,
    "cc1.4xlarge": 2,
    "cc2.8xlarge": 4,
    "cg1.4xlarge": 2,
    "cr1.8xlarge": 2,
    "d2.xlarge":   3,
    "d2.2xlarge":  6,
    "d2.4xlarge":  12,
    "d2.8xlarge":  24,
    "g2.2xlarge":  1,
    "g2.8xlarge":  2,
    "hi1.4xlarge": 2,
    "hs1.8xlarge": 24,
    "i2.xlarge":   1,
    "i2.2xlarge":  2,
    "i2.4xlarge":  4,
    "i2.8xlarge":  8,
    "m1.small":    1,
    "m1.medium":   1,
    "m1.large":    2,
    "m1.xlarge":   4,
    "m2.xlarge":   1,
    "m2.2xlarge":  1,
    "m2.4xlarge":  2,
    "m3.medium":   1,
    "m3.large":    1,
    "m3.xlarge":   2,
    "m3.2xlarge":  2,
    "m4.large":    0,
    "m4.xlarge":   0,
    "m4.2xlarge":  0,
    "m4.4xlarge":  0,
    "m4.10xlarge": 0,
    "r3.large":    1,
    "r3.xlarge":   1,
    "r3.2xlarge":  1,
    "r3.4xlarge":  1,
    "r3.8xlarge":  2,
}

def get_num_disks(instance_type):
    '''
    Given instance type, get number of instance disks for that instance
    '''
    if instance_type in _EC2_INSTANCE_NUM_DISKS:
        return _EC2_INSTANCE_NUM_DISKS[instance_type]
    else:
        return 0

def get_block_device_mapping(instance_type):
    '''
    Given instance type, return the block device mapping for m3 instances.
    We need to do this explicitly because AWS tends to ignore the
    because AMI-specified block device mapping.

    Later on, we could extend this functionality to allow customer to specify
    attaching external EBS volumes.
    '''
    block_map = BlockDeviceMapping()

    # AWS ignores the AMI-specified block device mapping for M3
    # See:
    # http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/block-device-mapping-concepts.html
    ### For M3 instances, you must specify instance store volumes in the block
    ### device mapping for the instance when you launch it. When you launch
    ### an M3 instance, instance store volumes specified in the block device
    ### mapping for the AMI may be ignored if they are not specified as part of
    ### the instance block device mapping.
    if instance_type.startswith('m3.'):
        for i in range(get_num_disks(instance_type)):
            dev = BlockDeviceType()
            dev.ephemeral_name = 'ephemeral%d' % i
            # The first ephemeral drive is /dev/sdb.
            name = '/dev/sd' + string.ascii_letters[i + 1]
            block_map[name] = dev
    return block_map
