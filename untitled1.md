features:
    ttl_log: logn(ttl) (so that big values dont overshadow small ones)
    s_acount: number of time the source address appears in the dataset
    rdatacount: number of different ip in the dataset for the q_name
    idcount: number of time the identifier appears in the dataset
    typecode: 0: A, 1: AAAA
    known: 1: in the whitelist
    name_len: length(q_name)
    name_lvl: number of levels in the domain name (www.google.com: 3)
    aa_flag
    tc_flag
    rd_flag
    ra_flag
    rcode
    answers_count: number of answer to the request
    authority_count
    additional_count