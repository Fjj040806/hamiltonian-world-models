"""Extract exact frames from archived scientific GIFs; no simulation or selection tuning."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json,hashlib
from matplotlib.font_manager import findfont
font_path=findfont("DejaVu Sans")
R=Path(__file__).resolve().parents[1];O=R/'paper/figures';O.mkdir(exist_ok=True)
font=ImageFont.truetype(font_path,24)
small=ImageFont.truetype(font_path,19)
records=[]
def strip(name,indices,labels,boxes,rowlabels,out):
 source=R/'phase7'/name;im=Image.open(source);cw=max(b[2]-b[0] for b in boxes);ch=max(b[3]-b[1] for b in boxes)
 sheet=Image.new('RGB',(cw*3+30,(ch+40)*len(boxes)+70),'white');d=ImageDraw.Draw(sheet)
 for col,(idx,label) in enumerate(zip(indices,labels)):
  im.seek(idx);frame=im.convert('RGB');d.text((col*cw+22,12),label,font=font,fill='#172f3e')
  for row,box in enumerate(boxes):
   crop=frame.crop(box);sheet.paste(crop,(col*cw+15,65+row*(ch+40)))
   if 'acquisition' in name and row==1:
    d.text((col*cw+160,65+row*(ch+40)+302),'q1',font=ImageFont.truetype(font_path,14),fill='black')
 for row,label in enumerate(rowlabels):d.text((20,40+row*(ch+40)),label,font=small,fill='#172f3e')
 sheet.save(O/out,dpi=(180,180))
 records.append({'figure':out,'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frame_indices':indices,'labels':labels,'crop_boxes':boxes,'row_labels':rowlabels,'interpretation':'exact GIF frame crops; no new numerical evaluation'})
strip('phase7_control_comparison.gif',[0,50,100],['t = 0','t = 10','t = 20'],[(65,438,380,630),(850,438,1165,630)],['Random, B = 64','Spectral A, B = 64'],'control_animation_filmstrip.png')
# Upper-right global A and lower-middle spectral A show the same variance scale.
strip('phase7_acquisition.gif',[0,16,64],['0 added labels','16 added labels','64 added labels'],[(750,80,1065,398),(425,408,740,710)],['Global A','Spectral A'],'acquisition_animation_filmstrip.png')
# A complete representative frame retains the common variance colorbar and context.
im=Image.open(R/'phase7/phase7_acquisition.gif');im.seek(32);im.convert('RGB').save(O/'acquisition_frame_32.png',dpi=(180,180))
records.append({'figure':'acquisition_frame_32.png','source':'phase7/phase7_acquisition.gif','frame_index':32,'interpretation':'complete unchanged frame at 32 acquired labels'})
(O/'animation_stills_provenance.json').write_text(json.dumps(records,indent=2))
print('Extracted three scientific figures from actual GIF frames.')
