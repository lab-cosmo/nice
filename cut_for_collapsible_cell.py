import json
import copy
import os

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

