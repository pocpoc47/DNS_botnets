#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', 'import modin.pandas as pd\nimport pandas as pdnm\nimport matplotlib.pyplot as plt\nimport time\nfrom sklearn.manifold import TSNE as sTSNE\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.preprocessing import RobustScaler')


# In[7]:


get_ipython().run_line_magic('time', '')
df = pd.read_csv('../2017_a3.csv')


# In[8]:


df['rdata']=df['rdata'].str.split('/')


# In[9]:


dtest = df[['q_name','rdata']]


# In[10]:


def tbyt(lis):
    for i in range(0,len(lis)):
        lis[i]=lis[i].split('.')[0]
    return lis


# In[11]:


dtest


# In[12]:


get_ipython().run_cell_magic('time', '', "\n#dtest = df[['q_name','rdata']]\ndtest['rdata'] = dtest['rdata'].dropna().apply(lambda x: tbyt(x))")


# In[ ]:





# In[13]:


get_ipython().run_cell_magic('time', '', "dtests = dtest.groupby('q_name').agg('sum')")


# In[14]:


get_ipython().run_cell_magic('time', '', 'dtestc = dtests[\'rdata\'].apply(lambda x: len(set(x)))\n#dtestcdic = dtestc.to_dict()\n#df[\'rdatacount\']=df[\'q_name\']\n#df = df.replace({\'rdatacount\':dtestcdic}).replace({\'rdatacount\':r"^.*\\..*$"},{\'rdatacount\':0},regex=True)\n\n#display(df)')


# In[16]:


dtestc.sort_values()


# In[ ]:


dtests.to_csv('../rdata.csv')


# In[17]:


dtestc.to_csv('../rdatacounts_first_byte.csv')


# In[63]:


rdatacounts = pd.read_csv('../rdatacounts.csv').set_index('q_name')
rdatacountsdic = rdatacounts['0'].to_dict()


# In[30]:


get_ipython().run_cell_magic('time', '', "df['rdatacount'] = df['q_name'].map(rdatacountsdic)")


# In[31]:


df.fillna({'rdatacount':0}, inplace=True)


# In[34]:


df.fillna({'rdatacount_first_byte':0}, inplace=True)


# In[32]:


df.fillna({'ttl':1}, inplace=True)


# In[ ]:


df.dropna(inplace=True,subset=['q_name'])


# In[8]:


whitelist3  = [
                'sophosxl.net',
                'fundp.ac.be',
                'unamur.be',
                'facebook.com',
                'google.com',
                'apple.com',
                'google.be',
                'fbcdn.net',
                'icloud.com',
                'gstatic.com',
                'youtube.com',
                'akamai.net',
                'discordapp.io',
                'outlook.com',
                'office.com',
                'dropbox.com',
                'live.com',
                'microsoft.com',
                'doubleclick.com',
                'office365.com',
                'twitter.com',
                'googleapis.com',
                'google-analytics.com',
                'akamaiedge.net',
                'googlesyndication.com',
                'akamaiedge.net',
                'doubleclick.net',
                'ytimg.com',
                'firefox.com',
                'facebook.net',
                'snapchat.com',
                'googleadservices.com',
                'akadns.net',
                'fbsbx.com',
                'digicert.com',
                'tmall.com',
                'qq.com',
                'baidu.com',
                'sohu.com',
                'login.tmall.com',
                'taobao.com',
                'jd.com',
                'yahoo.com',
                'wikipedia.org',
                'amazon.com',
                'sina.com.cn',
                'pages.tmall.com',
                'weibo.com',
                'zoom.us',
                'reddit.com',
                'netflix.com',
                'vk.com',
                'xinhuanet.com',
                'okezone.com',
                'csdn.net',
                'instagram.com',
                'alipay.com',
                'blogspot.com',
                'yahoo.co.jp',
                'twitch.tv',
                'myshopify.com',
                'bongacams.com',
                'google.com.hk',
                'bing.com',
                'microsoftonline.com',
                'tribunnews.com',
                'aliexpress.com',
                'stackoverflow.com',
                'naver.com',
                'panda.tv',
                'zhanqi.tv',
                'livejasmin.com',
                'babytree.com',
                'tianya.cn',
                'ebay.com',
                'amazon.co.jp',
                'google.co.in',
                'chaturbate.com']

for i in range(0,len(whitelist3)):
    whitelist3[i] = '(^|\.)'+whitelist3[i]+'$'


# In[ ]:


df['name_len'] = df['q_name'].str.len()


# In[ ]:


df['name_lvl']=df['q_name'].str.count('\.')+1


# In[9]:


get_ipython().run_line_magic('time', '')
df['known']= df['q_name'].str.lower().str.contains('|'.join(whitelist3)).astype(int)


# In[ ]:


types = df['q_type'].unique()


# In[ ]:


dic = {}
i=0
for t in types:
    dic[t] = i
    i+=1


# In[ ]:


df['typecode'] = df['q_type'].map(dic)


# In[ ]:


get_ipython().run_cell_magic('time', '', "df['idcount'] = df.groupby('identifier')['q_type'].transform('count')")


# In[ ]:


df['s_acount'] = df.groupby('s_addr')['q_type'].transform('count')


# In[ ]:


df['ttl_log']=np.log(df['ttl'])


# In[ ]:


df['ttl_log'] = df['ttl_log'].where(df['ttl_log']>=0,0)


# In[4]:


rdatacounts_fb = pd.read_csv('../rdatacounts_first_byte.csv').set_index('q_name')
rdatacountsdic_fb = rdatacounts_fb['rdata'].to_dict()


# In[5]:


get_ipython().run_cell_magic('time', '', "df['rdatacount_first_byte'] = df['q_name'].map(rdatacountsdic_fb)")


# In[6]:


df.fillna({'rdatacount_first_byte':0}, inplace=True)


# In[42]:


df.to_csv("../2017_a4.csv",index=False)


# In[28]:


df.fillna({'rdata':""},inplace=True)


# In[2]:


df = pd.read_csv("../2017_a4.csv")


# In[29]:


dfs = df.head(10000)


# In[34]:


def lnl(x):
    s=x.split(".")
    if(len(s)>1):
        return len(s[-2])
    else:
        return 0


# In[35]:


lnl("www.google.com")


# In[38]:


df['last_name_len'] = df['q_name'].dropna().apply(lnl)


# In[41]:


df[['last_name_len','q_name']].sort_values('last_name_len')


# In[ ]:


dd = df[['ttl_log','s_acount','rdatacount_fb','idcount','typecode','known','name_len','name_lvl','aa_flag','tc_flag','rd_flag','ra_flag','rcode','answers_count','authority_count','additional_count']]


# In[45]:


dd = df[(df['typecode']==0) | (df['typecode']==1)]
dd = df[['ttl_log','s_acount','rdatacount_first_byte','idcount','known','name_len','name_lvl','last_name_len']]


# In[ ]:


dd = dd[(dd['typecode']==0) | (dd['typecode']==1)]


# In[46]:


sample = dd.sample(n=1000)


# In[52]:


sample = dd


# In[ ]:


std = StandardScaler().fit_transform(sample.values)


# In[ ]:


rob = RobustScaler().fit_transform(sample.values)


# In[53]:


minmax = MinMaxScaler().fit_transform(sample.values)


# In[54]:


#weights
#'ttl_log','s_acount','rdatacount_first_byte','idcount','known','name_len','name_lvl','last_name_len'
w = np.array([
    3,
    2,
    6,
    3,
    6,
    3,
    3,
    6,
])


# In[ ]:


#weights
#'ttl_log','s_acount','rdatacount','idcount','typecode','known','name_len','name_lvl','aa_flag','tc_flag','rd_flag','ra_flag','rcode','answers_count','authority_count','additional_count'
w = np.array([
    5,
    5,
    6,
    3,
    1,
    6,
    4,
    3,
    1,
    1,
    1,
    1,
    1,
    3,
    1,
    1,
])


# In[ ]:


wstd = std*w


# In[ ]:


wrob = rob*w


# In[55]:


wminmax = minmax*w


# In[ ]:


kmeanstd = KMeans(n_clusters=5,random_state=1).fit(wstd)


# In[ ]:


kmeanrob = KMeans(n_clusters=5,random_state=1).fit(wrob)


# In[56]:


kmeanminmax = KMeans(n_clusters=5,random_state=1).fit(wminmax)


# In[57]:


dd.to_csv('temp.csv',index=0)


# In[58]:


ddc = pdnm.read_csv('temp.csv')


# In[59]:


ddc['cluster'] = pdnm.Series(kmeanminmax.labels_, index=ddc.index)


# In[60]:


ddc.groupby('cluster').mean()


# In[61]:


ddc[ddc['cluster']==1]


# In[ ]:



plt.figure(figsize=(20,10))
plt.plot(dd['q_type'].values, dd['name_len'].values, '.',markersize=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#ts = tsne.tsne(std[:1000],perplexity=60)\nts = sTSNE(perplexity=30).fit_transform(std)\nplt.figure(figsize=(20,10))\nplt.scatter(ts[:,0],ts[:,1], c=kmeanstd.labels_,cmap='viridis')\n\nplt.savefig('../tsne30-std-rdatacount_w.png')")


# In[ ]:


start = time.time()
#ts = tsne.tsne(rob[:1000],perplexity=60)
ts = sTSNE(perplexity=30).fit_transform(rob[:1000])
end = time.time()
print(end-start)
plt.figure(figsize=(20,10))
plt.scatter(ts[:,0],ts[:,1], c=kmeanrob.labels_,cmap='viridis')
plt.savefig('../tsne30-rob-rdatacount_w.png')


# In[51]:


start = time.time()
#ts = tsne.tsne(minmax[:1000],perplexity=60)
ts = sTSNE(perplexity=30).fit_transform(minmax[:1000])
end = time.time()
print(end-start)
plt.figure(figsize=(20,10))
plt.scatter(ts[:,0],ts[:,1], c=kmeanminmax.labels_,cmap='viridis')
plt.savefig('../tsne30-minmax-rdatacount_fb_lastname_w.png')


# In[ ]:





# In[ ]:




