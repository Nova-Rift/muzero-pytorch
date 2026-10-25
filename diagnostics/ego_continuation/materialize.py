"""Coordinator-only payload staging after exact storage admission; no GPU/network."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from run import digest, require

def validate(recipe):
    targets=set()
    for row in recipe['files']:
        target=Path(row['target']);source=Path(row['local'])
        require(not target.is_absolute() and '..' not in target.parts and str(target) not in targets,'Invalid/duplicate target')
        targets.add(str(target))
        require(source.is_file() and source.stat().st_size==row['bytes'] and digest(source)==row['sha256'],'Input changed: '+str(target))
    require(sum(r['bytes'] for r in recipe['files'])==recipe['total_bytes'],'Recipe byte total differs')

def stage(recipe_path,destination,archive=None):
    recipe=json.loads(recipe_path.read_text());validate(recipe)
    require(not destination.exists(),'Fresh stage directory required')
    require(shutil.disk_usage(destination.parent).free>recipe['total_bytes']*3+4*1024**3,'Insufficient stage/headroom bytes')
    destination.mkdir()
    for row in recipe['files']:
        out=destination/row['target'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['local'],out)
        require(digest(out)==row['sha256'],'Staged hash differs')
    manifest={'schema':1,'recipe_sha256':digest(recipe_path),'files':{r['target']:r['sha256'] for r in recipe['files']},'total_bytes':recipe['total_bytes']}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if archive:
        require(not archive.exists(),'Fresh archive required')
        with tarfile.open(archive,'w:gz') as tf:
            for path in sorted(destination.iterdir()):tf.add(path,arcname=path.name)
        manifest['archive']={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive)}
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--destination',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:validate(json.loads(a.recipe.read_text()));print('All recipe inputs verified; nothing staged.')
    else:
        require(a.destination is not None,'Destination required');print(json.dumps(stage(a.recipe,a.destination,a.archive),indent=2))
if __name__=='__main__':main()
