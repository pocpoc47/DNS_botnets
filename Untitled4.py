#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', 'import modin.pandas as pd\nimport pandas as pdnm\nimport matplotlib.pyplot as plt\nimport time\nimport tsne\nfrom sklearn.manifold import TSNE as sTSNE\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.preprocessing import RobustScaler')


# In[2]:


get_ipython().run_line_magic('time', '')
df = pd.read_csv('2017_a3.csv')


# In[3]:


df['rdata']=df['rdata'].str.split('/')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndtest = df[[\'q_name\',\'rdata\']]\ndtests = dtest.dropna().groupby(\'q_name\').agg(\'sum\')\ndtestc = dtests[\'rdata\'].apply(lambda x: len(set(x)))\n#dtestcdic = dtestc.to_dict()\n#df[\'rdatacount\']=df[\'q_name\']\n#df = df.replace({\'rdatacount\':dtestcdic}).replace({\'rdatacount\':r"^.*\\..*$"},{\'rdatacount\':0},regex=True)\n\n#display(df)')


# In[ ]:


dtests.to_csv('rdata.csv')


# In[ ]:


dtestc.to_csv('rdatacounts.csv')


# In[4]:


rdatacounts = pd.read_csv('rdatacounts.csv').set_index('q_name')
rdatacountsdic = rdatacounts['0'].to_dict()


# In[5]:


get_ipython().run_cell_magic('time', '', "df['rdatacount'] = df['q_name'].map(rdatacountsdic)")


# In[6]:


df.fillna({'rdatacount':0}, inplace=True)


# In[7]:


df.fillna({'ttl':1}, inplace=True)


# In[8]:


df.dropna(inplace=True,subset=['q_name'])


# In[13]:


whitelist3  = [
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


# In[10]:


df['name_len'] = df['q_name'].str.len()


# In[11]:


df['name_lvl']=df['q_name'].str.count('\.')+1


# In[14]:


get_ipython().run_line_magic('time', '')
df['known']= df['q_name'].str.lower().str.contains('|'.join(whitelist3)).astype(int)


# In[20]:


types = df['q_type'].unique()


# In[21]:


dic = {}
i=0
for t in types:
    dic[t] = i
    i+=1


# In[22]:


df['typecode'] = df['q_type'].map(dic)


# In[23]:


get_ipython().run_cell_magic('time', '', "df['idcount'] = df.groupby('identifier')['q_type'].transform('count')")


# In[24]:


df['s_acount'] = df.groupby('s_addr')['q_type'].transform('count')


# In[25]:


df['ttl_log']=np.log(df['ttl'])


# In[26]:


df['ttl_log'] = df['ttl_log'].where(df['ttl_log']>=0,0)


# In[29]:


df.to_csv("2017_a4.csv",index=False)


# In[ ]:


df.read_csv("2017_a4.csv")


# In[31]:


dd = df[['ttl_log','s_acount','rdatacount','idcount','typecode','known','name_len','name_lvl','aa_flag','tc_flag','rd_flag','ra_flag','rcode','answers_count','authority_count','additional_count']]


# In[33]:


dd = dd[(dd['typecode']==0) | (dd['typecode']==1)]


# In[52]:


sample = dd


# In[53]:


std = StandardScaler().fit_transform(sample.values)


# In[54]:


rob = RobustScaler().fit_transform(sample.values)


# In[55]:


minmax = MinMaxScaler().fit_transform(sample.values)


# In[109]:


#weights
#'ttl_log','s_acount','rdatacount','idcount','typecode','known','name_len','name_lvl','aa_flag','tc_flag','rd_flag','ra_flag','rcode','answers_count','authority_count','additional_count'
w = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
])


# In[110]:


wstd = std*w
wrob = rob*w
wminmax = minmax*w


# In[99]:


kmeanstd = KMeans(n_clusters=5,random_state=1).fit(wstd)


# In[100]:


kmeanrob = KMeans(n_clusters=5,random_state=1).fit(wrob)


# In[111]:


kmeanminmax = KMeans(n_clusters=5,random_state=1).fit(wminmax)


# In[59]:


dd.to_csv('temp.csv',index=0)


# In[60]:


ddc = pdnm.read_csv('temp.csv')


# In[112]:


ddc['cluster'] = pdnm.Series(kmeanminmax.labels_, index=ddc.index)


# In[108]:


ddc.groupby('cluster').mean()


# In[ ]:


plt.figure(figsize=(20,10))
ddc[ddc['cluster']==2]['ttl_log'].plot()


# In[ ]:



plt.figure(figsize=(20,10))
plt.plot(dd['q_type'].values, dd['name_len'].values, '.',markersize=3)


# In[49]:


get_ipython().run_cell_magic('time', '', "#ts = tsne.tsne(std[:1000],perplexity=60)\nts = sTSNE(perplexity=30).fit_transform(std)\nplt.figure(figsize=(20,10))\nplt.scatter(ts[:,0],ts[:,1], c=kmeanstd.labels_,cmap='viridis')\n\nplt.savefig('tsne30-std-rdatacount.png')")


# In[50]:


start = time.time()
#ts = tsne.tsne(rob[:1000],perplexity=60)
ts = sTSNE(perplexity=30).fit_transform(rob[:1000])
end = time.time()
print(end-start)
plt.figure(figsize=(20,10))
plt.scatter(ts[:,0],ts[:,1], c=kmeanrob.labels_,cmap='viridis')
plt.savefig('tsne30-rob-rdatacount.png')


# In[51]:


start = time.time()
#ts = tsne.tsne(minmax[:1000],perplexity=60)
ts = sTSNE(perplexity=30).fit_transform(minmax[:1000])
end = time.time()
print(end-start)
plt.figure(figsize=(20,10))
plt.scatter(ts[:,0],ts[:,1], c=kmeanminmax.labels_,cmap='viridis')
plt.savefig('tsne30-minmax-rdatacount.png')


# In[ ]:





# In[ ]:




