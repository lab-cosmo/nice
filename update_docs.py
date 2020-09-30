import os
os.system("rm -r ../build/*")
os.system("make html")
os.system("git checkout gh-pages")
os.system("cp -r ../build/* .")
with open(".nojekyll", "w") as f:
    pass
os.system("git add *")
os.system("git commit -m 'automatic docs build'")
os.system("git push")
os.system("git checkoug gh-pages")
