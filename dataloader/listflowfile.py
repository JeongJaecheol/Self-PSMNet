import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    all_right_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []
    test_right_disp = []
    filenames = [
        'flyingthings3d__frames_cleanpass',
        'driving__frames_cleanpass',
        'monkaa__frames_cleanpass',
        'flyingthings3d__frames_finalpass',
        'driving__frames_finalpass',
        'monkaa__frames_finalpass',
        'flyingthings3d__disparity',
        'driving__disparity',
        'monkaa__disparity'
    ]
    for filename in filenames:
        for (path, dir, files) in os.walk(filepath + filename + '/'):
            for filename_ in files:
                if path.find('TEST') > -1:
                    if path.find('left') > -1 and path.find('disparity') <= -1:
                        p = path + '/' + filename_
                        test_left_img.append(p)
                        test_right_img.append(p.replace('left', 'right'))
                        if 'driving__frames_cleanpass' in p:
                            tmp = p.replace("driving__frames_cleanpass", "driving__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'driving__frames_finalpass' in p:
                            tmp = p.replace("driving__frames_finalpass", "driving__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        elif 'flyingthings3d__frames_cleanpass' in p:
                            tmp = p.replace("flyingthings3d__frames_cleanpass", "flyingthings3d__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'flyingthings3d__frames_finalpass' in p:
                            tmp = p.replace("flyingthings3d__frames_finalpass", "flyingthings3d__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        elif 'monkaa__frames_cleanpass' in p:
                            tmp = p.replace("monkaa__frames_cleanpass", "monkaa__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'monkaa__frames_finalpass' in p:
                            tmp = p.replace("monkaa__frames_finalpass", "monkaa__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        test_left_disp.append(p.replace('.png', '.pfm'))
                        test_right_disp.append(p.replace('.png', '.pfm').replace('left', 'right'))
                else:
                    if path.find('left') > -1 and path.find('disparity') <= -1:
                        p = path + '/' + filename_
                        all_left_img.append(p)
                        all_right_img.append(p.replace('left', 'right'))
                        if 'driving__frames_cleanpass' in p:
                            tmp = p.replace("driving__frames_cleanpass", "driving__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'driving__frames_finalpass' in p:
                            tmp = p.replace("driving__frames_finalpass", "driving__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        elif 'flyingthings3d__frames_cleanpass' in p:
                            tmp = p.replace("flyingthings3d__frames_cleanpass", "flyingthings3d__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'flyingthings3d__frames_finalpass' in p:
                            tmp = p.replace("flyingthings3d__frames_finalpass", "flyingthings3d__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        elif 'monkaa__frames_cleanpass' in p:
                            tmp = p.replace("monkaa__frames_cleanpass", "monkaa__disparity")
                            p = tmp.replace("frames_cleanpass", "disparity")
                        elif 'monkaa__frames_finalpass' in p:
                            tmp = p.replace("monkaa__frames_finalpass", "monkaa__disparity")
                            p = tmp.replace("frames_finalpass", "disparity")
                        all_left_disp.append(p.replace('.png', '.pfm'))
                        all_right_disp.append(p.replace('.png', '.pfm').replace('left', 'right'))

    return all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp
''' 
 classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
 image = [img for img in classes if img.find('frames_cleanpass') > -1]
 disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

 monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
 monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
 
 monkaa_dir  = os.listdir(monkaa_path)

 all_left_img=[]
 all_right_img=[]
 all_left_disp = []
 test_left_img=[]
 test_right_img=[]
 test_left_disp = []


 for dd in monkaa_dir:
   for dd2 in os.listdir(monkaa_path+'/'+dd+'/'):
     for im in os.listdir(monkaa_path+'/'+dd+'/'+dd2+'/left/'):
      if is_image_file(monkaa_path+'/'+dd+'/'+dd2+'/left/'+im):
       all_left_img.append(monkaa_path+'/'+dd+'/'+dd2+'/left/'+im)
       all_left_disp.append(monkaa_disp+'/'+dd+'/'+dd2+'/left/'+im.split(".")[0]+'.pfm')

     for im in os.listdir(monkaa_path+'/'+dd+'/'+dd2+'/right/'):
      if is_image_file(monkaa_path+'/'+dd+'/'+dd2+'/right/'+im):
       all_right_img.append(monkaa_path+'/'+dd+'/'+dd2+'/right/'+im)
  
 flying_path = filepath + [x for x in image if x.find('frames_cleanpass') > -1][0]
 flying_disp = filepath + [x for x in disp if x.find('disparity') > -1][0]
 flying_dir = flying_path+'/TRAIN/'
 subdir = ['A','B','C']

 for ss in subdir:
    flying = os.listdir(flying_dir+ss)

    for ff in flying:
      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      for im in imm_l:
       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

 flying_dir = flying_path+'/TEST/'

 subdir = ['A','B','C']

 for ss in subdir:
    flying = os.listdir(flying_dir+ss)

    for ff in flying:
      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      for im in imm_l:
       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)



 driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
 driving_disp = filepath + [x for x in disp if 'driving' in x][0]

 subdir1 = ['15mm_focallength','15mm_focallength']
 subdir2 = ['scene_backwards','scene_forwards']
 subdir3 = ['fast','slow']

 for i in subdir1:
   for j in subdir2:
    for k in subdir3:
        imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
        for im in imm_l:
          if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
            all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
          all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

          if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
            all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp
'''

