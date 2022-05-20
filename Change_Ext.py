import os
import cv2
rename='jpg'
bad_images=[]
bad_ext=[]
for subdir, dirs, files in os.walk(r'C:\Python_Content\People_2'):
    for filename in files:
        filepath = subdir + os.sep + filename
        #Get the extension and remove . from it
        base_file, ext = os.path.splitext(filepath)
        ext = ext.replace('.','')   
        print (filepath)
        if ext in rename:
            #Create the new file name
            new_ext = rename
            new_file = base_file + '.' + 'jpeg'
            #Create the full old and new path
            old_path = os.path.join(r'C:\Users\meima\Downloads\People', filepath)
            new_path = os.path.join(r'C:\Users\meima\Downloads\People', new_file)

            #Rename the file
            os.rename(old_path, new_path)
        
if len(bad_images) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_images)):
        os.remove(bad_images[i])
      
else:
    print(' no improper image files were found')
            


'''
if os.path.isfile(filepath):
            try:
                img=cv2.imread(filepath)
                shape=img.shape
            except:
                print('file ', filepath, ' is not a valid image file')
                bad_images.append(filepath)
        else:
            print('*** fatal error, you a sub directory  in class directory ')
'''