"""Download a TGIF trained model together with its paired preprocessing statistics."""
import argparse,json,shutil,tempfile,urllib.request,zipfile
from pathlib import Path
from pipeline.common import sha256

def unpack(archive,output,seed):
    prefix=f'seed_{seed}/'
    names={prefix+name for name in ['model.pt','normalization.npz','metadata.json','sha256.json']}
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if {m.filename for m in members}!=names or len(members)!=4:
            raise ValueError('Unexpected archive contents')
        if sum(m.file_size for m in members)>64*1024*1024:
            raise ValueError('Model archive is too large')
        if any(m.is_dir() or (m.external_attr>>16)&0o170000==0o120000 for m in members):
            raise ValueError('Archive links/directories are not accepted')
        z.extractall(output)
    return Path(output)/f'seed_{seed}'

def download(seed,output):
    manifest=json.loads(Path(__file__).with_name('models.json').read_text())
    entry=manifest['seeds'][str(seed)]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/f'seed_{seed}'
    if target.exists():
        sums=json.loads((target/'sha256.json').read_text())
        if all(sha256(target/name)==value for name,value in sums.items()):
            print('Already verified:',target);return target
        raise ValueError('Existing model bundle is incomplete or modified; choose a clean output directory')
    with tempfile.TemporaryDirectory(dir=output) as temp:
        temp=Path(temp);archive=temp/'weights.zip'
        with urllib.request.urlopen(entry['url'],timeout=60) as r,open(archive,'wb') as f:
            shutil.copyfileobj(r,f)
        if sha256(archive)!=entry['sha256']:raise ValueError('Downloaded archive checksum mismatch')
        bundle=unpack(archive,temp,seed)
        sums=json.loads((bundle/'sha256.json').read_text())
        if not all(sha256(bundle/name)==value for name,value in sums.items()):raise ValueError('Extracted bundle checksum mismatch')
        shutil.move(str(bundle),str(target))
    print('Downloaded and verified:',target);return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[41,17,43,5,31,59],default=43)
    p.add_argument('--output',default=str(Path(__file__).with_name('checkpoints')))
    a=p.parse_args();download(a.seed,a.output)
