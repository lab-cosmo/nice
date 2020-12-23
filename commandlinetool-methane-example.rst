Introduction
============
There are two command line tools. The first one takes inputs to fit a model using the NICE sequence, and the second one takes this fitted NICE model as input to give predicted NICE features. An example is given below using a 'methane dataset <https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528>', that is saved as methane.extxyz in our folder:
::
wget "https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528" -O methane.extxyz.gz
gunzip -k methane.extxyz.gz