import urllib.request, os, sys

os.makedirs('/app/models/pnlcalib', exist_ok=True)
files = {
    'SV_kp': 'https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp',
    'SV_lines': 'https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines'
}
for name, url in files.items():
    dest = f'/app/models/pnlcalib/{name}'
    print(f'Downloading {name}...')
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f'{name}: {size/1024/1024:.1f}MB')
    assert size > 10_000_000, f'{name} too small — download failed'
print('PnLCalib weights OK')
