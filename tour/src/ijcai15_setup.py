#!/usr/bin/env python3
import requests
import tempfile
import zipfile
import os
import sys

data_url = 'https://sites.google.com/site/limkwanhui/datacode/data-ijcai15.zip?attredirects=0'
chunk_size = 4096
data_dir = '../data'
fname = None

if os.path.exists(data_dir) == False:
    print('Data directory "' + data_dir + '" not found,')
    print('Please create it.')
    sys.exit(0)

# download data (zip file) to a temp file
print('Download data from', data_url)
r = requests.get(data_url, stream=True)
with tempfile.NamedTemporaryFile(delete=False) as fd:
    fname = fd.name
    for chunk in r.iter_content(chunk_size):
        fd.write(chunk)

# unzip: output is a directory 'data-ijcai15' in data_dir
assert(zipfile.is_zipfile(fname))
zipfile.ZipFile(fname).extractall(path=data_dir)

# delete the temp file
os.unlink(fname)

print('Data available at directory', os.path.join(data_dir, 'data-ijcai15'))
