import os
import json
import copy

# cutting notebooks
def split(name, destination):
    with open("tutorials/{}".format(name), "r") as f:
        notebook = json.load(f)
        
    before_collapsible = copy.deepcopy(notebook)
    before_collapsible['cells'] = before_collapsible['cells'][0:2]

    collapsible = copy.deepcopy(notebook)
    collapsible['cells'] = [collapsible['cells'][2]]

    after_collapsible = copy.deepcopy(notebook)
    after_collapsible['cells'] = after_collapsible['cells'][3:]
    
    clean_name = name.strip().split('.')[0]
    with open("{}/{}_before_collapsible.ipynb".format(destination,
                                                      clean_name), "w") as f:
        json.dump(before_collapsible, f)
    
    with open("{}/{}_collapsible.ipynb".format(destination,
                                                      clean_name), "w") as f:
        json.dump(collapsible, f)


    with open("{}/{}_after_collapsible.ipynb".format(destination,
                                                      clean_name), "w") as f:
        json.dump(after_collapsible, f)
        
os.system("mkdir docs/cutted")       
split('calculating_covariants.ipynb', 'docs/cutted/')
split('getting_insights_about_the_model.ipynb', 'docs/cutted/')
split('constructor_or_non_standard_sequence.ipynb', 'docs/cutted/')
split('sequential_fitting.ipynb', 'docs/cutted/')
split('custom_regressors_into_purifiers.ipynb', 'docs/cutted/')

# converting notebooks to rst 

def make_substitution(lines, index):
    lines_before = lines[0:index]
    end = len(lines)
    for j in range(index + 1, len(lines)):
        if not(lines[j].strip() == "" or lines[j].startswith('    ')):
            end = j
            break

    lines_raw = lines[index + 1 : end]
    raw_from = 0
    for i in range(len(lines_raw)):
        if (lines_raw[i].strip() != ''):
            raw_from = i
            break
    lines_raw = lines_raw[raw_from:]
    
    raw_to = 0
    for i in range(len(lines_raw)):
        if (lines_raw[i].strip() != ''):
            raw_to = i
            
    lines_raw = lines_raw[:raw_to]

    
    lines_for_insertion = [".. raw:: html\n",
                           "\n",
                           "<embed>\n",
                           "<pre>\n",
                           '<p style="margin-left: 5%;font-size:12px;line-height: 1.2" >\n']
    lines_for_insertion = lines_for_insertion + lines_raw     
    lines_for_insertion = lines_for_insertion + ["</p>\n", "</pre>\n", "</embed>\n", '\n']
                       
    for i in range(1, len(lines_for_insertion)):
        lines_for_insertion[i] = '    ' + lines_for_insertion[i]
        
    return lines_before + lines_for_insertion + lines[end:]   
    
    return lines[index : end]    
        
def get_bad_block(lines):   
    for i in range(len(lines)):
        if (lines[i].strip() == ".. parsed-literal::"):
            return i
    return None

def iterate(lines):
    while True:
        index = get_bad_block(lines)
        if index is None:
            return lines
        lines = make_substitution(lines, index)
    
def fix_awful_nvconvert_format(file):
    lines = []
    with open(file, "r") as f:
        lines = list(f)
    lines = iterate(lines)
    with open(file, "w") as f:
        for line in lines:
            f.write(line)

os.chdir('docs/cutted/')
names = [name for name in os.listdir('.') if name.endswith('.ipynb')]

for name in names:
    dir_name = name.split('.')[0]    
    os.system("mkdir {}".format(dir_name))
    os.system("cp {} {}/".format(name, dir_name))
    os.chdir(dir_name)
    os.system('jupyter nbconvert --to rst {}'.format(name))
    fix_awful_nvconvert_format(name.split('.')[0] + '.rst')
    names_inner = os.listdir('.')
    for name_inner in names_inner:
        if (name_inner.endswith('_files')):  
            os.system('cp -r {} ../../'.format(name_inner))  
    os.chdir('../')

    
os.chdir('../..')

os.system("rm -r ../build/*")
os.chdir("./docs")
os.system("sphinx-apidoc -f -o . ../nice")
os.system("make html")
os.chdir("../")
os.system("git checkout -f gh-pages")
os.system("git rm -r *")
os.system("cp -r ../build/html/* .")
with open(".nojekyll", "w") as f:
    pass

os.system("git add *")
os.system("git add .nojekyll")
os.system("git commit -m 'automatic docs build'")
os.system("git push")
os.system("git checkout master")

