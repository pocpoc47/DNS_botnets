We tweaked the dns extraction script so that we could extract the dns ttl, the source and destination ip address

The features we are using as of today(12/06) are as follows:

'name_len','q_type',idcount,'answers_count', 'aa_flag','tc_flag','rd_flag','ra_flag','rcode','ttl','answers_count','authority_count','additional_count','known'

Where
    -name_len is the length of the q_name
    -idcount is the number time the identifier appears in the dataset
    -known is 1 if the q_name ends with one of the domains in whitelist consisting of the top50 domains on alexa alongside known domains that were appearing the most in the dataset

We dropped opcode and questions_count since those features were the same for all datapoints.
We currently don't use the q_name, only its length and its presence or not in our whitelist

We used KMeans for clustering after applying StandardScaler().fit_transform().

We used t-SNE to reduce the dimensionality for the visualisations.

The visualisation was done with a subset of 1000 datapoints to reduce load.

We found some interesting patterns in the data:
    -many DNS rr have a ttl close to the maximum (4294967295) (136 years)
    -identifiers were not equally distributed, much more request had the 1234 identifier, and most of those were asking for a.root-servers.net
    -two ip addresses appeared much more than others as source address:
    
        -> 187.124.15.148      496909 (this is the DNS server and is therefore normal)
        -> 187.124.43.215     1087937 (this address is doing an unreasonable amount of requests, especially on various domains of the unamur)

test
