"""Restore all bundled datasets/checkpoints and verify every extracted file."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
M=json.loads((R/'artifacts/bundle_manifest.json').read_text())
for part in M['parts']:
 p=R/'artifacts'/part['file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256'],p
 with zipfile.ZipFile(p) as z:
  for name in z.namelist():
   target=(R/name).resolve()
   if not target.is_relative_to(R):raise ValueError('Invalid archive path')
  z.extractall(R)
for name,sha in M['restored_files'].items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
print(f"Restored and verified {len(M['restored_files'])} datasets/checkpoints.")
