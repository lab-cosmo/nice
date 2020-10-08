import os
os.chdir('docs/cutted/')
names = [name for name in os.listdir('.') if name.endswith('.ipynb')]

for name in names:
    dir_name = name.split('.')[0]    
    os.system("mkdir {}".format(dir_name))
    os.system("cp {} {}/".format(name, dir_name))
    os.chdir(dir_name)
    os.system('jupyter nbconvert --to rst {}'.format(name))
    os.chdir('../')
    
os.chdir('../..')