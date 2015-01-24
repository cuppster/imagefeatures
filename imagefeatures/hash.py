
def gen_signed_hash(obj):
    from cityhash import CityHash64
    return CityHash64(obj)

def gen_hex_hash(obj):
    from cityhash import CityHash64
    return hex(CityHash64(obj))[2:-1]

def gen_hash(obj):
    from cityhash import CityHash64
    return unsigned2signed64(CityHash64(obj))

def unsigned2signed64(i):
    return (i + 2**63) % 2**64 - 2**63

def signed2unsigned64(i):
    return i % 2**64


# deprecated

def gen_owner_agent_hash(owner, agent):
    return gen_hash('{0}/{1}'.format(owner, agent))

def gen_owner_agent_fid_hash(owner, agent, fid):
    return gen_hash('{0}/{1}/{2}'.format(owner, agent, fid))
